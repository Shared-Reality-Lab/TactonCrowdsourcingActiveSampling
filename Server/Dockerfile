FROM fastgenomics/pandas:0.22-p36-v3
RUN apk update && apk add postgresql-dev gcc python3-dev musl-dev
RUN pip3 install flask flask_sqlalchemy psycopg2 Werkzeug scipy
RUN pip3 install gunicorn
WORKDIR /app
ADD . /app

#COPY /usr/share/dict/words /usr/share/dict/words

CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app"]
#CMD  [ "python3", "./main.py" ]