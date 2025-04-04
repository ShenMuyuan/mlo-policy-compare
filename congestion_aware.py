from hol_model import HOL_Model
import numpy as np

def congestion_aware_beta(nmld, nsld1, nsld2, beta_range, tt1, tt2, tf1, tf2, lam1, lam21, lam22, e2edelay_max = 1000000):
    # e2edelay_max过滤掉超过这个delay的beta值
    beta_valid_range = []
    err_beta = []
    for beta in beta_range:
        model1 = HOL_Model(
            n1 = nmld,
            n2 = nsld1,
            lambda1 = tt1 * lam1 * beta,
            lambda2 = tt1 * lam21,
            W_1 = 16,
            W_2 = 16,
            K_1 = 6,
            K_2 = 6,
            tt = tt1,
            tf = tf1
            )
        model2 = HOL_Model(
            n1 = nmld,
            n2 = nsld2,
            lambda1 = tt2 * lam1 * (1 - beta),
            lambda2 = tt2 * lam22,
            W_1 = 16,
            W_2 = 16,
            K_1 = 6,
            K_2 = 6,
            tt = tt2,
            tf = tf2
        )
        if model1.state[0] == "U" and model2.state[0] == "U" and model1.e2e_delay[0] < e2edelay_max and model2.e2e_delay[0] < e2edelay_max:
            beta_valid_range.append(beta)
            alpha1, alpha2 = model1.alpha, model2.alpha
            err_beta.append( (beta - alpha1 / (alpha1 + alpha2)) ** 2 )
    congestion_idx = np.argmin(err_beta)
    return beta_valid_range[congestion_idx]