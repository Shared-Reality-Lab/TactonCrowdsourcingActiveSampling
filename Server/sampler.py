import numpy as np
import pandas as pd
import random
import itertools

import itertools
from collections import defaultdict

np.set_printoptions(suppress=True, precision=4, threshold=np.inf, linewidth=np.inf)


class Sampler():
    """base class for sampler object"""

    def _sample(self, fonction, length, already_pressed, user_uuid, db):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._sample(*args, **kwargs)


class RandomSampler(Sampler):
    def _sample(self, fonction, length, already_pressed, user_uuid, db):
        if int(fonction) == 0:
            samples = [random.randint(0, 1) * 255 for _ in range(length)]
        else:
            samples = [random.randint(0, 255) for _ in range(length)]

        return samples


class EIGSampler(Sampler):
    """ the same N pairs are presented to each user, and the graph grows the same way as above for the other users
    However this is done in a Batch fashion instead of pairwise.
    Uses EIG MST when sampling."""

    def __init__(self, learner):
        self.learner = learner

        random.seed(0)

        self.same_samples_list = None  # initialized later

        self.samples_list = [[255 * random.randint(0, 1) for _ in range(20)] for _ in range(2000)]

        starter_kit_per_user = []
        for _ in range(10):
            starter_kit_per_user.append(random.choice(self.samples_list))

        self.samples_per_user = defaultdict(lambda: starter_kit_per_user)
        self.samples_grouped = defaultdict(list)

        self.global_iter = 0
        self.epsilon = 0.2

        self.tacton_buffer = []
        self.mst_iters = 0

    @staticmethod
    def _get_probabilities_sampling(user_count):
        n = tuple()
        if user_count < 25:
            n = (0.2, 0.6)
        elif user_count < 50:
            n = (0.15, (1 + 0.15) / 2)
        elif user_count < 75:
            n = (0.125, (1 + 0.125) / 2)
        else:
            n = (0.10, (1 + 0.10) / 2)
        return n

    def _get_exploration_percent(self):
        min_epsilon = 0.1
        decay = 0.99
        epsilon = max(min_epsilon, self.epsilon * decay)
        self.epsilon = epsilon
        return epsilon

    def _sample(self, fonction, length, user_uuid, num_buttons, db):
        """ samples all the bunch at the same time and returns the N samples"""
        # count now starts at 0.
        user_count = pd.read_sql("""select count(uuid) from answers where uuid = '{}'""".format(user_uuid),
                                 db.engine).values[0].item()
        # print(user_count)
        min_num_clusters = 2

        # Every nth pair, up to a given number of repeats, send the same pairs across all users
        if user_count in [2, 4, 6]:
            if self.same_samples_list is None:  # first iteration
                self.same_samples_list = [[255 * random.randint(0, 1) for _ in range(20)] for _ in range(num_buttons)]
            samples = self.same_samples_list
            random.shuffle(samples)
            print("COMMON")
            is_group = True
            return samples, min_num_clusters, is_group
        else:
            if self.global_iter == 0:
                samples = [[255 * random.randint(0, 1) for _ in range(20)] for _ in range(num_buttons + 4)]
                from main import add_samples_to_db
                ids = add_samples_to_db(samples)
                self.learner.add_new_tactons(ids)
                samples = random.sample(samples, num_buttons - 1)
            else:  # after 1st global iteration
                if len(self.tacton_buffer) < (num_buttons - 1) or self.mst_iters > 5:
                    self.mst_iters = 0
                    self.tacton_buffer = []
                    tacton_ids = self.learner.forward()

                    joined_tacton_ids = ",".join([str(id) for id in
                                                  tacton_ids])
                    pats_dict = pd.read_sql(
                        """select index, data from patterns where index in ({})""".format(joined_tacton_ids),
                        db.engine).set_index("index").to_dict(orient="index")

                    pats = []
                    for t_id in tacton_ids:
                        pat = pats_dict[t_id]["data"]
                        pats.append(pat)

                    formatted_patterns = [xx.strip("[").strip("]").split(",") for xx in pats]
                    formatted_patterns = [[int(xx) for xx in x] for x in formatted_patterns]

                    self.tacton_buffer.extend(
                        formatted_patterns[num_buttons - 2:])  # the last one will always be the attn test

                    samples = formatted_patterns[:num_buttons - 2]
                    to_add = random.sample(self.samples_list, 1)  # pick way more to add, and then cut
                    samples = samples + to_add
                else:
                    samples = self.tacton_buffer[:num_buttons - 1]  # the last one will always be the attn test
                    self.tacton_buffer = self.tacton_buffer[num_buttons - 1:]

                self.mst_iters += 1

            # Attn test ---> replace one sample at random with another from the same batch
            random_idxs = random.sample(list(range(len(samples))), 1)
            samples.append(samples[random_idxs[0]])
            random.shuffle(samples)

            self.samples_per_user[user_uuid].extend(samples)
            self.samples_grouped[user_uuid].append(samples)

            self.global_iter += 1
            is_group = False
            return samples, min_num_clusters, is_group
