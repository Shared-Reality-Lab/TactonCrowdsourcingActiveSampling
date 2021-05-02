import numpy as np

x_gh, w_gh = np.polynomial.hermite.hermgauss(20)
x_gh = x_gh.reshape(-1, 1)
w_gh = w_gh.reshape(-1, 1)
eps = 1e-8


def model_probabilities(pcm):
    """this code is from zhi li, sureal package"""
    assert pcm.shape[0] == pcm.shape[1]
    assert pcm.ndim == 2

    mu = np.exp(pcm) / (np.exp(pcm) + np.exp(pcm.T))
    var = pcm * pcm.T / ((pcm + pcm.T) ** 2)

    return mu, var


def to_hyperbolic(x):
    """from range [0, 1] to [-1, 1]"""
    return x * 2 - 1


def active_learning_pairwise(mus, vars):
    mean = to_hyperbolic(mus)
    std = np.sqrt(vars)
    return _calculate_eig(mean, std)


def _calculate_eig(mu, sigma):
    M, M_ = np.shape(sigma)

    sigma = np.reshape(sigma, (1, -1))
    mu = mu.reshape(*sigma.shape)  # np.zeros(sigma.shape)

    exp_y = lambda x: np.exp(-(np.sqrt(2) * sigma * x + mu))
    s_ij = lambda x: (1 + exp_y(x)) ** (-1)
    pi_cnst = np.pi ** (-0.5)

    f1 = lambda y: s_ij(y) * np.log(s_ij(y)) * pi_cnst
    f2 = lambda y: s_ij(y) * pi_cnst
    f3 = lambda y: (1 - s_ij(y)) * np.log(1 - s_ij(y)) * pi_cnst
    f4 = lambda y: (1 - s_ij(y)) * pi_cnst

    es1 = (w_gh * f1(x_gh)).sum(0)
    es2 = (w_gh * f2(x_gh)).sum(0)
    es3 = (w_gh * f3(x_gh)).sum(0)
    es4 = (w_gh * f4(x_gh)).sum(0)

    ig = es1 - es2 * np.log(es2 + eps) + es3 - es4 * np.log(es4 + eps)
    return ig.reshape((M, M_))


if __name__ == "__main__":
    import random

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    np.set_printoptions(suppress=True, precision=4, threshold=np.inf, linewidth=np.inf)

    init_size_ = 10  # equivalent to starter_kit_per_user.
    alpha_test = np.ones((init_size_, init_size_))
    counts = np.zeros((init_size_, init_size_))
    DIAG_FILL_VALUE = 1.
    np.fill_diagonal(alpha_test, DIAG_FILL_VALUE)

    epsilon = 0.10
    min_epsilon = 0.05
    decay = 0.99
    J = 0
    UNWEIGHTED_PCMS = []
    for i in range(int(1. * 1000 - 1)):
        UNWEIGHTED_PCMS.append(alpha_test)
        mu, cova = model_probabilities(alpha_test)

        eig_matrix = active_learning_pairwise(mu, cova)

        assert not np.all(np.isnan(eig_matrix))

        # eig_matrix *= ((counts+counts.T)<10).astype(int)

        from scipy.sparse.csgraph import minimum_spanning_tree

        tcsr = minimum_spanning_tree(-np.triu(eig_matrix))
        tcsr_tmp = tcsr.toarray()
        pairs = np.argwhere(tcsr_tmp < 0)
        np.random.shuffle(pairs)

        num_to_keep = 5
        to_keep = []
        for row in pairs:
            i1, i2 = row
            # if i1 in to_keep or i2 in to_keep:
            #     continue
            if not i1 in to_keep:
                to_keep.append(i1)
            # if len(to_keep) >= num_to_keep: break
            if not i2 in to_keep:
                to_keep.append(i2)
            # if len(to_keep) >= num_to_keep: break

        for k, idx in enumerate(range(0, len(to_keep), num_to_keep)):

            tactons_to_sample = to_keep[idx:idx + num_to_keep]  # + add_random_one#.tolist()
            # print(tactons_to_sample)
            if len(tactons_to_sample) < num_to_keep or k >= 10:
                break

            rdm_tacton = random.sample(tactons_to_sample, 1)
            tactons_to_sample = tactons_to_sample + rdm_tacton
            random.shuffle(tactons_to_sample)

            import itertools


            def _maybe_swap(tacton1, tacton2):
                if tacton1 > tacton2:
                    tacton1, tacton2 = tacton2, tacton1
                return tacton1, tacton2


            #### gather answers
            J += 1

            # weighted_sampling_list = [-1] * 50 + [0] * 30 + [1] * 50
            weighted_sampling_list = [1, 4, 0, 3, 3, 2, 1, 4, 2, 0]
            answers = random.sample(weighted_sampling_list, len(tactons_to_sample))
            # answers = [random.randint(-1, 1) for _ in range(len(tactons_to_sample))]
            ans_w_indexes = list(zip(answers, tactons_to_sample))

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

                first_tacton_id, second_tacton_id = _maybe_swap(first_tacton_id, second_tacton_id)

                counts[first_tacton_id, second_tacton_id] += 1

                if first_tacton_grp == second_tacton_grp and first_tacton_grp != -1 and second_tacton_grp != -1:  # same group
                    # alpha_test[first_tacton_id, second_tacton_id] += 3
                    tacton_list_similar.append((first_tacton_id, second_tacton_id))
                else:  # diff group
                    # alpha_test[second_tacton_id, first_tacton_id] += 1
                    tacton_list_dissimilar.append((first_tacton_id, second_tacton_id))

            assert eta == 0.5 or eta == 1

            total_num_grps_formed = max([x[0] for x in ans_w_indexes]) + 1
            for tacton1, tacton2 in tacton_list_similar:
                sim_w = total_num_grps_formed / len(tacton_list_similar)
                alpha_test[tacton1, tacton2] += sim_w  # * eta
                # alpha_test[tacton1, tacton2] += 1
            for tacton1, tacton2 in tacton_list_dissimilar:
                dissim_w = total_num_grps_formed / len(tacton_list_dissimilar)
                alpha_test[tacton2, tacton1] += dissim_w  # * eta
                # alpha_test[tacton2, tacton1] += 1

            # UNWEIGHTED_PCMS.append(alpha_test)

        # epsilon = max(min_epsilon, epsilon*decay)
        # if random.random() < epsilon:
        if random.random() < 0.10:
            # if (i+1) % 10 == 0: #add new tacton in the mix
            fill_value = 1.
            alpha_test = np.append(alpha_test, [[fill_value] * len(alpha_test.T)], 0)
            alpha_test = np.append(alpha_test, [[fill_value]] * len(alpha_test), 1)

            # alpha_test = np.append([[fill_value] * len(alpha_test.T)], alpha_test , 0)
            # alpha_test = np.append([[fill_value]] * len(alpha_test), alpha_test, 1)

            np.fill_diagonal(alpha_test, DIAG_FILL_VALUE)

            counts = np.append(counts, [[0.] * len(counts.T)], 0)
            counts = np.append(counts, [[0.]] * len(counts), 1)

            # counts = np.append([[0.] * len(counts.T)], counts, 0)
            # counts = np.append([[0.]] * len(counts), counts, 1)

    import matplotlib.pyplot as plt

    renormalized_alpha = alpha_test - DIAG_FILL_VALUE
    print("**" * 10)
    print(renormalized_alpha)
    print("done")

    ## distribution of values
    # dist = []
    # for i in range(alpha_test.shape[0]):
    #     for j in range(alpha_test.shape[1]):
    #         exponential_score = np.exp(alpha_test[i, j]) / (np.exp(alpha_test[i, j]) + np.exp(alpha_test[j, i]))
    #         if not exponential_score in [0.5]:
    #             dist.append(exponential_score)
    # plt.hist(dist, bins=20, cumulative=False)
    # plt.show()
    print("Ratio similarity/dissimilarity:", np.triu(alpha_test, 1).sum() / np.tril(alpha_test, 1).sum())

    import seaborn as sns

    print("average upper triangle count", counts[np.triu_indices(counts.shape[0], 1)].mean())
    print("J={}".format(J))
    sns.heatmap(np.clip(counts, 0, None), cmap="coolwarm");
    plt.show()

    np.save("/home/marc/PycharmProjects/Haptrix-Analysis/counts.npy", counts)
    np.save("/home/marc/PycharmProjects/Haptrix-Analysis/alpha_test.npy", alpha_test)

    # sns.heatmap(renormalized_alpha)
    # plt.show()
    #
    probability_mtx = np.exp(renormalized_alpha) / (np.exp(renormalized_alpha) + np.exp(renormalized_alpha.T))
    #
    sns.heatmap(np.triu(probability_mtx, 1),
                cmap="RdBu",
                mask=np.tril(np.ones(probability_mtx.shape)))
    plt.show()

    # Animation

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    UNWEIGHTED_PCMS_DOWNSAMPLED = UNWEIGHTED_PCMS[30::10]


    def update(i):

        vmin = (UNWEIGHTED_PCMS_DOWNSAMPLED[i]).min()
        vmax = (UNWEIGHTED_PCMS_DOWNSAMPLED[i]).max()
        g = sns.heatmap(UNWEIGHTED_PCMS_DOWNSAMPLED[i] - 1., vmin=vmin, vmax=vmax,
                        cmap="Reds", ax=ax, cbar=False, annot=False,
                        xticklabels=False, yticklabels=False)
        plt.tight_layout()


    anim = FuncAnimation(fig, update, frames=len(UNWEIGHTED_PCMS_DOWNSAMPLED), interval=50)
    anim.save("test_animation.gif", writer="imagemagick", dpi=200, fps=10.0)
    plt.show()
