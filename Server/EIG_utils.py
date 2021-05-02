import numpy as np
import math
from scipy import linalg
import sys

GAUSSIAN_HERMITE_N = 30

def paired_comparisons(pcm, link_fcn='probit'):
    M, N = np.shape(pcm)
    X = np.zeros((int(N * (N - 1) / 2), N))

    weights = np.zeros((int(N * (N - 1) / 2), 1))
    counts = np.zeros_like(weights)

    # generate the model matrix X and the response vector Y (weights and counts)
    k = 0
    for mm in range(0, M - 1):
        for nn in range(mm + 1, M):
            weights[k] = pcm[mm, nn]
            counts[k] = pcm[mm, nn] + pcm[nn, mm]
            X[k, mm] = 1
            X[k, nn] = -1
            k += 1

    dist_y = 'binomial'  # distribution of the responses y
    import statsmodels
    try:
        b, deviance, resid_degrees_freedom, covariance = fit_glm(X, np.concatenate([weights, counts - weights], axis=1),
                                                                 dist_y, link_fcn, constant='off')  # counts-weights
    except statsmodels.tools.sm_exceptions.PerfectSeparationError:
        b = np.zeros((N, 1))
        covb = np.zeros((N, N))
        for i in range(covb.shape[0]):
            for j in range(covb.shape[1]):
                if i != j:
                    covb[i, j] = 0.25
                else:
                    covb[i, j] = 0.5

        return b, None, None, covb

    from scipy.stats import chi2
    pfit = chi2.sf(deviance, resid_degrees_freedom)  # df is degrees of freedom in "stats"
    if pfit < 0.05:
        print("fit is suspect, pvalue=", pfit)
        print("For {} degrees of freedom the deviance should be less than {}".format(resid_degrees_freedom,
                                                                                     chi2.ppf(0.95,
                                                                                              resid_degrees_freedom)))

    return b, resid_degrees_freedom, pfit, covariance

def fit_glm(X, y, dist_y, link_fcn, constant='off'):
    import statsmodels.api as sm

    ## set up GLM
    # y = np.concatenate((y, np.ones([len(y), 1])), axis=1)
    sm_probit_Link = sm.genmod.families.links.probit
    sm_probit_Link = sm.genmod.families.links.logit
    sm_logit_link = sm.genmod.families.links.logit()
    glm_binom = sm.GLM(y, X,#sm.add_constant(V_design_matrix),
                       family=sm.families.Binomial(link=sm_logit_link))
    # statsmodels.GLM format: glm_binom = sm.GLM(data.endog, data.exog, family)

    ## Run GLM fit
    glm_result = glm_binom.fit()
    weights_py = glm_result.params
    covariance = glm_result.cov_params()
    return weights_py, glm_result.deviance, glm_result.df_resid, covariance

def run_my_modeling_BT(alpha):
    """ my version"""

    b_prior, df, pfit, covariance = paired_comparisons(alpha, 'logit')

    # for i in range(1, Number_stimuli - 1):
    #     for j  in range(i+1, Number_stimuli):
    #         mu(i, j) = mu_v(i) - mu_v(j);
    #         sigma(i, j) = sqrt(covb(i, i) + covb(j, j) - 2. * covb(i, j));
    #         eig(label) = Gaussian_Hermite_BT(mu(i, j), sigma(i, j), n);
    #         index(label,:) = [i, j];
    return b_prior.ravel(), covariance

def run_modeling_Bradley_Terry(alpha):
    """this code is from zhi li, sureal package"""

    # alpha = np.array(
    #     [[0, 3, 2, 7],
    #      [1, 0, 6, 3],
    #      [4, 3, 0, 0],
    #      [1, 2, 5, 0]]
    #     )

    M, M_ = alpha.shape
    assert M == M_

    iteration = 0
    p = 1.0 / M * np.ones(M)
    change = sys.float_info.max

    DELTA_THR = 1e-8
    n = alpha + alpha.T

    while change > DELTA_THR:
        iteration += 1
        p_prev = p
        pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
        p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1) # summing over axis=1 marginalizes j

        p = p / np.sum(p)

        change = linalg.norm(p - p_prev)

    pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
    # lbda_ii = np.sum(-alpha / np.tile(p, (M, 1)) ** 2 + n / pp ** 2, axis=1) #todo orig
    lbda_ii = np.sum(-alpha / np.tile(p, (M, 1)).T ** 2 + n / pp ** 2, axis=1)

    lbda_ij = n / pp * 2
    lbda = lbda_ij + np.diag(lbda_ii)
    cova = np.linalg.pinv(
        np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])])
    )
    vari = np.diagonal(cova)[:-1]
    # print("vari", vari)
    stdv = np.sqrt(vari)

    scores = np.log(p)
    # print("scores", scores)

    scores_std = stdv / p  # y = log(x) -> dy = 1/x * dx
    assert np.all(p>=0)
    assert np.all(p<=1)
    return scores, cova[:-1, :-1], scores_std  # CI = scores +- 1.96 * scores_std  (1.96 for Z(alpha/2) where alpha = 0.05


def EIG_GaussianHermitte_matrix_Hybrid_MST(mu_mtx, sigma_mtx):
    """ this is the matrix implementation version"""
    """mu is the matrix of difference of two means (si-sj), sigma is the matrix of sigma of si-sj"""
    epsilon = 1e-9
    M, M_ = np.shape(mu_mtx)

    mu = np.reshape(mu_mtx, (1, -1))
    sigma = np.reshape(sigma_mtx, (1, -1))

    fs1 = lambda x: (1. / (1 + np.exp(-np.sqrt(2) * sigma * x - mu))) * (
        -np.log(1 + np.exp(-np.sqrt(2) * sigma * x - mu))) / np.sqrt(math.pi);
    fs2 = lambda x: (1 - 1. / (1 + np.exp(-np.sqrt(2) * sigma * x - mu))) * (
        np.log(np.exp(-np.sqrt(2) * sigma * x - mu) / (1 + np.exp(-np.sqrt(2) * sigma * x - mu)))) / np.sqrt(math.pi);
    fs3 = lambda x: 1. / (1 + np.exp(-np.sqrt(2) * sigma * x - mu)) / np.sqrt(math.pi);
    fs4 = lambda x: (1 - 1. / (1 + np.exp(-np.sqrt(2) * sigma * x - mu))) / np.sqrt(math.pi);

    import numpy.polynomial.hermite as herm
    x, w = herm.hermgauss(GAUSSIAN_HERMITE_N)
    x = np.reshape(x, (-1, 1))
    w = np.reshape(w, (-1, 1))

    es1 = np.sum(w * fs1(x), 0)
    es2 = np.sum(w * fs2(x), 0)
    es3 = np.sum(w * fs3(x), 0)
    es3 = es3 * np.log(es3 + epsilon)
    es4 = np.sum(w * fs4(x), 0)
    es4 = es4 * np.log(es4 + epsilon)

    # ret = es1 + es2 - es3 + es4 #TODO there was an error here in orig code
    ret = es1 + es2 - es3 - es4
    ret = np.reshape(ret, (M, M_))
    # ret = -np.triu(ret, 1) #TODO modified
    return ret# + ret.T


def ActiveLearningPair_matrix_Hybrid_MST(mu, mu_cova):
    pvs_num = len(mu)

    eig = np.zeros((pvs_num, pvs_num))

    mu_1 = np.tile(mu, (pvs_num, 1))

    sigma = np.diag(mu_cova)
    sigma_1 = np.tile(sigma, (pvs_num, 1))

    mu_diff = mu_1.T - mu_1

    sigma_diff = np.sqrt(sigma_1.T + sigma_1 - 2 * mu_cova)
    eig = EIG_GaussianHermitte_matrix_Hybrid_MST(mu_diff, sigma_diff)
    return eig

if __name__ == "__main__":
    import random
    random.seed(0)
    np.random.seed(0)
    np.set_printoptions(suppress=True)
    # alpha_test = np.array(
    #         [[0, 2, 1, 1],
    #          [1, 0, 5, 1],
    #          [1, 1, 0, 10],
    #          [5, 1, 1, 0]]
    #         )
    alpha_test = np.array(
        [[0, 10, 6, 8],
         [6, 0, 10, 10],
         [10, 10, 0, 8],
         [8, 8, 10, 0]]
    )
    alpha_test = np.array(
        [[0, 2, 1, 1],
         [1, 0, 1, 1],
         [2, 2, 0, 2],
         [2, 2, 1, 0]]
    )
    alpha_test = np.array(
        [[0, 3, 2, 1],
         [1, 0, 2, 1],
         [1, 1, 0, 2],
         [1, 1, 1, 0]]
    )
    # alpha_test = np.ones((4, 4), dtype=np.float) * 1
    # np.fill_diagonal(alpha_test, 0.)
    # alpha_test[2, 3] = 3.
    # alpha_test[1, 3] = 5.
    # alpha_test = np.array(
    #     [[0, 1, 1, 1],
    #      [1, 0, 1, 1],
    #      [1, 1, 0, 1],
    #      [1, 1, 1, 0]]
    #     )

    for i in range(60):
        print("iter i=", i)
        print(alpha_test)
        v=[]
        for j in range(alpha_test.shape[0]):
            for k in range(j, alpha_test.shape[1]):
                if j != k:
                    v.append(alpha_test[j, k] + alpha_test[k, j])
        print(v)

        score, cova, std = run_modeling_Bradley_Terry(alpha_test)
        print(np.exp(score))
        print(np.argsort(-np.exp(score)))
        # score, cova = run_my_modeling_BT(alpha_test)
        print(score)
        # print(cova)
        # print(std)

        eig_matrix = ActiveLearningPair_matrix_Hybrid_MST(score, cova)
        assert not np.all(np.isnan(eig_matrix))
        # print("*"*10)
        print(np.isnan(eig_matrix).any())

        # print(np.triu(eig_matrix))
        if np.isnan(eig_matrix).any():
            print("yes")

        EIG_mtx_triu = -np.triu(eig_matrix, 1)
        np.fill_diagonal(EIG_mtx_triu, 0.)
        pairsBT = np.unravel_index(np.argsort(EIG_mtx_triu.ravel()), EIG_mtx_triu.shape)
        pairs = np.array(pairsBT).T


        from scipy.sparse.csgraph import minimum_spanning_tree
        if alpha_test.sum() > np.prod(alpha_test.shape):
            tcsr = minimum_spanning_tree(-np.triu(eig_matrix))
            tcsr_tmp = tcsr.toarray()
            pairMST = np.where(tcsr_tmp < 0)
            pairs = np.array(pairMST).T

        # remove lower triangle
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]


        for p in pairs[:5]:
            if random.random() < 0.3:
                alpha_test[p[0], p[1]] += 1
                # alpha_test[p[1], p[0]] += 1
            # else:
            #     alpha_test[p[1], p[0]] += 1

        if i % 15 == 0:
            fill_value = 1.0# np.amax(alpha_test)
            # alpha_test = np.append(alpha_test, [[fill_value] * len(alpha_test.T)], 0)
            # alpha_test = np.append(alpha_test, [[fill_value]] * len(alpha_test), 1)

            alpha_test = np.append([[fill_value] * len(alpha_test.T)], alpha_test , 0)
            alpha_test = np.append([[fill_value]] * len(alpha_test), alpha_test, 1)

            np.fill_diagonal(alpha_test, 0.)

    print("**"*10)
    print(alpha_test-1.)
    print("done")