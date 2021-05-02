import os

import numpy as np

np.set_printoptions(precision=2)
from EIG_utils_similarity import model_probabilities, active_learning_pairwise
import itertools
from datetime import datetime
from scipy.sparse.csgraph import minimum_spanning_tree

GAUSSIAN_HERMITE_N = 30


class EIGLearner:
    INIT_FILL_VALUE = 1.
    DIAG_FILL_VALUE = 1.
    COUNT_FILL_VALUE = 0.

    def __init__(self, pcm=np.array([[]])):
        self.pcm = pcm
        self.row_matcher = {}  # matches rows & columns in the pcm with samples
        self.unweighted_pcm = np.zeros_like(pcm)
        self.exp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.pairs_buffer = []

    def forward(self, MST=True):
        """simulate pair comparison procedure for single learning based on kl divergence"""
        if self.pcm.size == 0:
            # generate 10 random tactons
            return list(range(10))

        mu, cova = model_probabilities(self.pcm)

        EIG_mtx = active_learning_pairwise(mu, cova)

        if MST:
            tcsr = minimum_spanning_tree(-np.triu(EIG_mtx))
            pairMST = np.where(tcsr.A < 0)
            pairs = np.array(pairMST).T
            np.random.shuffle(pairs)
        else:
            EIG_mtx_triu = -np.triu(EIG_mtx, 1)
            pairsBT = np.unravel_index(np.argsort(EIG_mtx_triu.ravel()), EIG_mtx_triu.shape)
            pairs = np.array(pairsBT).T

        # from the np.array of pairs, find the most likely pairs of hapids to be compared
        def f(x):
            p = dict(zip(self.row_matcher.values(), self.row_matcher.keys()))
            return p.get(x)

        def map_row_col_to_hapid(x):
            return np.array([np.fromiter((f(xxi) for xxi in xi), x.dtype) for xi in x])

        def map_row_col_to_hapid_raveled(x):
            return [self.row_matcher[xx] for xx in x]

        # remove lower triangle
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]

        # map them to actual hapids using the row_matcher
        pp = map_row_col_to_hapid(pairs)

        # flatten the pairs
        p = pp.ravel()
        _, idx = np.unique(p, return_index=True)
        to_keep = p[np.sort(idx)].tolist()
        return to_keep

    def _save_pcm(self):
        os.makedirs("./logs", exist_ok=True)
        np.save("./logs/exp_{}_pcm_{}".format(self.exp_id, int(self.pcm.sum())), arr=self.pcm, allow_pickle=False)
        np.save("./logs/exp_{}_mapper_{}".format(self.exp_id, int(self.pcm.sum())), arr=self.row_matcher,
                allow_pickle=True)
        np.save("./logs/exp_{}_unweighted_pcm_{}".format(self.exp_id, int(self.pcm.sum())), arr=self.unweighted_pcm,
                allow_pickle=False)

    def update_pcm(self, answer_list, hapid_list):
        assert isinstance(answer_list, list)
        assert isinstance(answer_list[0], int)
        assert isinstance(hapid_list, list)
        assert isinstance(hapid_list[0], int)

        def _maybe_swap(tacton1, tacton2):
            if tacton1 > tacton2:
                tacton1, tacton2 = tacton2, tacton1
            return tacton1, tacton2

        ans_w_indexes = list(zip(answer_list, hapid_list))

        tacton_list_similar, tacton_list_dissimilar = [], []
        for combination in itertools.combinations(sorted(ans_w_indexes), 2):
            cc = list(combination)
            first_tacton_grp, first_tacton_id = cc[0]
            second_tacton_grp, second_tacton_id = cc[1]

            # if the two tacton_id's compared are the same, it's an attention test and it serves us as a proxy to rate the annotator weight $\eta_k$
            if first_tacton_id == second_tacton_id:
                eta = int(
                    first_tacton_grp == second_tacton_grp and first_tacton_grp >= 0 and second_tacton_grp >= 0)
                if not eta: eta = 0.5

                continue  # skip rest of steps

            first_tacton_id, second_tacton_id = _maybe_swap(self.row_matcher[first_tacton_id],
                                                            self.row_matcher[second_tacton_id])

            if first_tacton_grp == second_tacton_grp and first_tacton_grp != -1 and second_tacton_grp != -1:  # same group
                tacton_list_similar.append((first_tacton_id, second_tacton_id))
            else:  # diff group
                tacton_list_dissimilar.append((first_tacton_id, second_tacton_id))

        assert eta == 0.5 or eta == 1

        total_num_grps_formed = max([x[0] for x in ans_w_indexes]) + 1
        for tacton1, tacton2 in tacton_list_similar:
            sim_w = total_num_grps_formed / len(tacton_list_similar)
            self.pcm[tacton1, tacton2] += sim_w * eta
            self.unweighted_pcm[tacton1, tacton2] += 1

        for tacton1, tacton2 in tacton_list_dissimilar:
            dissim_w = total_num_grps_formed / len(tacton_list_dissimilar)
            self.pcm[tacton2, tacton1] += dissim_w * eta
            self.unweighted_pcm[tacton2, tacton1] += 1

        np.fill_diagonal(self.pcm, self.DIAG_FILL_VALUE)
        self._save_pcm()

    def add_new_tactons(self, tactons_list):
        return self._add_new_rows(tactons_list)

    def _get_last_row_number(self):
        try:
            max_key = max(list(self.row_matcher.values())) + 1
        except ValueError:
            max_key = 0
        return max_key

    def _shift_row_matcher(self):
        for k in self.row_matcher:
            self.row_matcher[k] += 1

    def _add_new_rows(self, tacton_ids):
        max_key = self._get_last_row_number()
        for tacton_id in tacton_ids:
            if tacton_id not in self.row_matcher.keys():

                self.row_matcher[tacton_id] = max_key
                if not self.pcm.size == 0:
                    self.pcm = np.append(self.pcm, [[self.INIT_FILL_VALUE] * len(self.pcm.T)], 0)
                    self.pcm = np.append(self.pcm, [[self.INIT_FILL_VALUE]] * len(self.pcm), 1)

                    self.unweighted_pcm = np.append(self.unweighted_pcm,
                                                    [[self.COUNT_FILL_VALUE] * len(self.unweighted_pcm.T)], 0)
                    self.unweighted_pcm = np.append(self.unweighted_pcm,
                                                    [[self.COUNT_FILL_VALUE]] * len(self.unweighted_pcm), 1)
                else:
                    self.pcm = np.append(self.pcm, [[self.INIT_FILL_VALUE]], 1)

                    self.unweighted_pcm = np.append(self.unweighted_pcm, [[self.COUNT_FILL_VALUE]], 1)

                max_key += 1
        np.fill_diagonal(self.pcm, self.DIAG_FILL_VALUE)


if __name__ == "__main__":
    test_pcm = np.array([[1, 2, 3], [0, 8, 6], [4, 5, 3]])
    test_learner = EIGLearner(test_pcm)
    pairs_test = test_learner.forward()
    print(pairs_test)
