#!/usr/local/bin/python

from flask import Flask
from flask import request
from flask import json
from flask_sqlalchemy import SQLAlchemy
import random
import hashlib
import pandas as pd
import logging
import time
from sqlalchemy import MetaData
import os
import uuid
from sampler import BunchClusterSampler, EIGSampler
from EIG import EIGLearner
from threading import Lock
import string

pd.set_option('display.max_columns', 10)

type = os.getenv('DB_TYPE', 'postgresql')
dbuser = os.getenv('DB_USER', 'haptrix')
dbpass = os.getenv('DB_PASSWORD', 'haptrixcim')
dburl = os.getenv('DB_HOST', 'localhost')
dbport = os.getenv('DB_PORT', '5432')
dbname = os.getenv('DB_NAME', 'haptrix')
port = os.getenv('PORT', '5000')
host = os.getenv('HOST', '127.0.0.1')
debug = eval(os.getenv('DEBUG', "False"))

url = type + '://' + dbuser + ':' + dbpass + '@' + dburl + ':' + dbport + '/' + dbname
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = debug  # debug database records

db = SQLAlchemy(app)

active_learner = EIGLearner()
sampler = EIGSampler(active_learner)

lock = Lock()

desired_characters = "".join([x for x in string.ascii_uppercase if x not in ['S', 'O', 'I']])
desired_digits = "".join([x for x in string.digits if x not in ['0', '1', '5']])

MAX_COUNTER = 8  # For batch mode, this is the total number of batches
NUM_CALIBRATIONS = 0
NUM_VIBRATIONS = 6  # For batch mode, number of vibrations
assert NUM_VIBRATIONS <= 12, "we don't want to overwhelm the participants!"
APP_VERSIONS_TO_ACCEPT = [31]
APP_VERSION = 31

APP_PASSCODE = "see"


def gensig_cluster(fonction, length, user_uuid, num_buttons):
    """ first_signal_sent : for when you have pairwise comparison and you want to prevent the same signal to be compared twice"""
    global db

    samples, min_num_clusters, is_group = sampler(fonction, length, user_uuid, num_buttons, db)

    ids = add_samples_to_db(samples)

    return samples, ids, min_num_clusters, is_group


def add_samples_to_db(samples):
    ids = []
    with db.engine.connect() as conn, conn.begin():
        if conn.dialect.has_table(db.engine, 'patterns'):
            app.logger.debug('table exist')
            meta = MetaData()
            meta.reflect(bind=db.engine)  # , reflect=True)
            table = meta.tables['patterns']
            for sample in samples:
                previous = pd.read_sql("""select * from patterns where data = '{}'""".format(sample), db.engine)
                if len(previous) == 1:
                    id = previous.iloc[0]["index"]
                elif len(previous) > 1:
                    print("multipleresultsfound")
                    app.logger.debug('More than one identical pattern found in DB')
                    id = previous.iloc[0]["index"]
                else:
                    app.logger.debug('new combinaison: ' + str(sample))
                    id = pd.read_sql("""select max(index) from patterns;""", db.engine).values[0].item()
                    id += 1
                    newdata = pd.DataFrame(data={'data': str(sample)}, index=[id])
                    newdata.to_sql('patterns', db.engine, if_exists='append')
                ids.append(id)
        else:
            app.logger.debug('no table')
            for idx, sample in enumerate(samples):
                if idx != 0:
                    previous = pd.read_sql("""select * from patterns where data = '{}'""".format(sample), db.engine)
                    if len(previous) == 1:
                        id = previous.iloc[0]["index"]
                    elif len(previous) > 1:
                        print("multipleresultsfound")
                        app.logger.debug('More than one identical pattern found in DB')
                        id = previous.iloc[0]["index"]
                    else:
                        app.logger.debug('new combinaison: ' + str(sample))
                        id = pd.read_sql("""select max(index) from patterns;""", db.engine).values[0].item()
                        id += 1
                        newdata = pd.DataFrame(data={'data': str(sample)}, index=[id])
                        newdata.to_sql('patterns', db.engine, if_exists='append')
                else:
                    id = 0
                    app.logger.debug('new combinaison: ' + str(sample))
                    newdata = pd.DataFrame(data={'data': str(sample)}, index=[id])
                    newdata.to_sql('patterns', db.engine, if_exists='append')
                ids.append(id)
    return ids


def checkpattern_batch(ids, chksum):
    global db
    test = False
    check = hashlib.blake2s(str(ids).encode('utf-8')).hexdigest()
    if check == chksum:
        test = True
    return test


def generate_completion_token():
    token = ''.join(random.choice(desired_characters + desired_digits) for _ in range(5))
    return token


def saveanswer(uuid, hapid, question, answer, time_taken, button_pressed_sequence, is_group):
    with db.engine.connect() as conn, conn.begin():
        # the data is set in a list to avoid the pandas error: you must set an index if passing scalar values
        newdata = pd.DataFrame(data={'uuid': uuid,
                                     'hapid': str(hapid),
                                     'question': question,
                                     'answer': answer,
                                     'time_taken': time_taken,
                                     'button_sequence': str(button_pressed_sequence),
                                     'is_calibration': False}, index=[time.time()])

        newdata.to_sql('answers', db.engine, if_exists='append', index_label='save time')

        user_s_ratings_count = pd.read_sql(
            """select count(hapid) as total from answers where uuid = '{}' group by uuid;""".format(uuid),
            db.engine).iloc[0, 0]

        token = generate_completion_token()

        token_table = pd.DataFrame(data={'uuid': uuid,
                                         'completion_token': token,
                                         'user_rating_count': user_s_ratings_count}, index=[time.time()])

        token_table.to_sql('tokens', db.engine, if_exists='append', index_label='save time')

        answer_ints = [int(x) for x in answer.strip("[").strip("]").split(",")]
        hapid_ints = [int(x) for x in hapid.strip("[").strip("]").split(",")]

        if not is_group:
            active_learner.update_pcm(answer_ints, hapid_ints)

    return user_s_ratings_count, token


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/app_info/')
def get_version_and_password():
    json_object = json.jsonify(
        app_version=str(APP_VERSION),
        app_passcode=str(APP_PASSCODE),
        num_vibrations=str(NUM_VIBRATIONS),
    )
    return json_object


def generate_n_signals(length_signal, total_length, fonction):
    fonction = int(fonction)
    if fonction == 0:
        return [[random.randint(0, 1) * 255 for _ in range(length_signal)] for _ in range(total_length)]
    elif fonction == 1:
        return [[random.randint(0, 255) for _ in range(length_signal)] for _ in range(total_length)]


@app.route('/pattern/<user_uuid>/<fonction>/<int:num>/<int:num_buttons>/<int:user_rating_count>')
def generate_cluster_patterns(user_uuid, fonction, num=20, num_buttons=10, user_rating_count=-1):
    """ function or pairwise ranking"""
    with lock:

        user_app_version = int(user_uuid.split("_")[-1])

        if not user_app_version in APP_VERSIONS_TO_ACCEPT:
            return json.jsonify({})

        questions_list = ['Similar?']

        # hapsig = generate_n_signals(num, num_buttons, fonction)
        question_asked = random.sample(questions_list, 1)[0]
        # hapids = list(range(num_buttons)) #TODO this needs to be sent to the database!

        hapsig2, hapids, min_num_clusters, is_group = gensig_cluster(fonction, num, user_uuid, num_buttons)
        if not is_group:
            active_learner.add_new_tactons(hapids)
        hapids = str(hapids)

        check = hashlib.blake2s(hapids.encode('utf-8')).hexdigest()
        json_object = json.jsonify(
            hapsig=hapsig2,
            hapid=hapids,
            checksum=check,
            question=question_asked,
            min_num_clusters=min_num_clusters,
            is_group=is_group
        )
    return json_object


@app.route('/answer', methods=['POST'])
def answer():
    with lock:
        hapids = request.form["hapid"]
        test = checkpattern_batch(hapids, request.form['checksum'])
        if test:

            is_group = request.form["is_group"] == "True" or request.form["is_group"] == "true"
            user_ratings_count, completion_token = saveanswer(request.form['UUID'],
                                                              request.form['hapid'],
                                                              request.form['question'],
                                                              request.form['answer'],
                                                              request.form['time_taken'],
                                                              request.form['button_pressed_sequence'],
                                                              is_group)

            if user_ratings_count >= MAX_COUNTER:  # max number of ratings a particular user can give
                user_ratings_count = -1

            json_obj = json.jsonify(
                status='saved',
                user_rating_count=str(user_ratings_count),
                max_count=str((MAX_COUNTER - NUM_CALIBRATIONS)),
                code=completion_token,
                calibration_answer=str(0),
                total_num_calibrations=str(NUM_CALIBRATIONS),
                calibration_justification=""
            )
        else:
            app.logger.debug('Error in pattern matching with checksum')
            json_obj = json.jsonify(
                status='not found'
            )
    return json_obj


@app.route('/id/<marque>/<modele>/<amplitude>/<user_uuid>', methods=['GET'])
def create_user_uuid(marque, modele, amplitude, user_uuid):
    global db
    with db.engine.connect() as conn, conn.begin():
        app.logger.debug('New uuid requested')
        # uuid_ = uuid.uuid4()
        newdata = pd.DataFrame(data={'uuid': str(user_uuid),
                                     'brand': marque,
                                     'model': modele,
                                     'amplitude': amplitude,
                                     # 'email': mail
                                     }, index=[time.time()])
        newdata.to_sql('users', db.engine, if_exists='append', index_label='save time')
    return json.jsonify(uuid=user_uuid)


@app.route('/counter/<user_uuid>', methods=['GET'])
def get_current_uuid_counter(user_uuid):
    global db
    with db.engine.connect() as conn, conn.begin():

        try:
            res = pd.read_sql("""select user_rating_count, completion_token from tokens where
            user_rating_count = (
                select max(user_rating_count)
                from tokens where uuid = '{}')
            and uuid = '{}'""".format(user_uuid, user_uuid), db.engine).iloc[0, :]
        except:  # if user is not registered yet
            res = (None, "no ratings given!")

        if int(user_uuid.split("_")[1]) not in APP_VERSIONS_TO_ACCEPT:
            return json.jsonify(count=0, last_code_given="no ratings given!", max_count=str(MAX_COUNTER),
                                total_num_calibrations=NUM_CALIBRATIONS)

        counter, last_code_given = res[0], res[1]

        if counter is None:  # case where the user still hasn't given a rating
            counter = 0

        if counter >= MAX_COUNTER:
            counter = -1

        # We need to return the max counter to make sure that we display the right count on screen
        return json.jsonify(count=str(counter),
                            last_code_given=last_code_given,
                            max_count=str(MAX_COUNTER - NUM_CALIBRATIONS),
                            total_num_calibrations=NUM_CALIBRATIONS
                            )


@app.route('/user', methods=['POST'])
def create_user_uuid_entry():
    if request.method == 'POST':

        user_login = request.form['user_login']
        user_password = request.form['user_password']

        with db.engine.connect() as conn, conn.begin():
            if conn.dialect.has_table(db.engine, 'users'):
                app.logger.debug('table users exists')
                meta = MetaData()
                meta.reflect(bind=db.engine)  # , reflect=True)
                table = meta.tables['users']

                # TODO return user id corresponding to the login here - this could be made better with better knowledge of sqlalchemy and using the table
                # TODO needs protection against errors
                list_users = db.session.execute("SELECT * FROM users").fetchall()
                uuid_ = None  # initialize to None
                for (uuid__, login) in list_users:
                    if login == user_login:
                        uuid_ = uuid__

                if uuid_ is None:  # generate a uuid, because it wasnt in the database
                    app.logger.debug('user not already present in table, making new uuid')
                    # make new user id (random uuid = uuid4)
                    uuid_ = uuid.uuid4()
                    newdata = pd.DataFrame(data={'uuid': [str(uuid_)],
                                                 'id': [str(user_login)]})
                    newdata.to_sql('users', db.engine, if_exists='append', index=False)
            else:
                app.logger.debug('no table `users` in database')
                # TODO in this case create table and add user to it

    else:
        app.logger.error('Corrupted user login')

    return str(uuid_)


if __name__ == '__main__':
    app.run(debug=debug, port=port, host=host)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
