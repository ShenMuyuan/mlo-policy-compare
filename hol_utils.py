from scipy.optimize import fsolve
from typing import Tuple
import numpy as np


# 节点饱和时的服务率 mu_S
def calc_mu_S(p: float, tt: float, tf: float, W: int, K: int, alpha=None):
    if alpha is None:
        alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    ret = (
            2
            * alpha
            * tt
            * p
            * (2 * p - 1)
            / (2 * p - 1 + W * (p - 2 ** K * (1 - p) ** (K + 1)))
    )
    return ret


# 节点未饱和时的服务率 mu_U
def calc_mu_U(p: float, tt: float, tf: float, W: int, K: int, lambda1: float, alpha: float):
    mu_S = calc_mu_S(p, tt, tf, W, K, alpha)
    ret = mu_S / (1 + (1 - tf / tt * (1 - p) / p) * (mu_S - lambda1))
    return ret


def calc_alpha_sym(tt, tf, n, p):
    alpha = 1 / (
            tf + 1 + (tt - tf) * (n * p - n * p ** (n / (n - 1))) - tf * p ** (n / (n - 1))
    )
    return alpha


def calc_alpha_base(tt, tf, p):
    alpha = 1 / (1 + tf - tf * p - (tt - tf) * p * np.log(p))
    return alpha


def calc_alpha_asym(tt, tf, n1, p1, n2, p2):
    n = n1 + n2
    pp = (p1 ** n1 * p2 ** n2) ** (1 / (n - 1))
    alpha = 1 / (tf + 1 + (tt - tf) * (n1 * p1 + n2 * p2 - n * pp) - tf * pp)
    return alpha


def relu(x):
    return x if x > 0 else 0


def calc_conf(p1, p2, lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, alpha, state_idx):
    # UU: 0   US: 1   SU:2   SS: 3
    if (p1 < 0 or p2 < 0):  # SMY: ignore prob < 0 case
        return -np.inf
    mu1 = calc_mu_S(p1, tt, tf, W_1, K_1, alpha)
    mu2 = calc_mu_S(p2, tt, tf, W_2, K_2, alpha)
    if mu1 < 0 or mu2 < 0:
        return 0
    state_signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return 1 - 0.5 * (
            relu(state_signs[state_idx][0] * (lambda1 - mu1))
            + relu(state_signs[state_idx][1] * (lambda2 - mu2))
    )


# based on calc_au_p_fsolve (let one n = 0)
def calc_unsat_p_fsolve(
        n, lamb, tt, tf
):
    def pf(p):
        p_ans = [0]
        alpha = calc_alpha_sym(tt, tf, n, p[0])
        p_ans[0] = p[0] - (1 - lamb / (alpha * tt * p[0])) ** (n - 1)
        return p_ans

    p_u = fsolve(pf, [0.9])
    unsat = True
    if (not np.isclose(pf(p_u), [0.0])) or p_u[0] > 1:
        unsat = False
    return p_u[0], unsat


# p
def calc_au_p_fsolve(
        n1: int, lambda1: float, n2: int, lambda2: float, tt: float, tf: float
):
    def pf(p, n_1, n_2, lambda_1, lambda_2):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_1, p[0], n_2, p[1])
        p_ans[0] = p[0] - (1 - lambda_1 / (alpha * tt * p[0])) ** (n_1 - 1) * \
                   (1 - lambda_2 / (alpha * tt * p[1])) ** n_2
        p_ans[1] = p[1] - (1 - lambda_1 / (alpha * tt * p[0])) ** n_1 * \
                   (1 - lambda_2 / (alpha * tt * p[1])) ** (n_2 - 1)
        return p_ans

    p_au = fsolve(pf, [0.9, 0.9], args=(n1, n2, lambda1, lambda2))
    err = np.sqrt(np.sum(np.array(pf(p_au, n1, n2, lambda1, lambda2)) ** 2))
    uu = True
    if err > 1e-5 or p_au[0] > 1 or p_au[1] > 1:
        uu = False
    return p_au, uu


def calc_ps_p_fsolve(
        n1: int,
        lambda1: float,
        n2: int,
        lambda2: float,
        W_1: int,
        K_1: int,
        W_2: int,
        K_2: int,
        tt: float,
        tf: float,
):
    def psf(p, n_u, n_s, W_s, K_s, lambda_u):
        p_ans = [0, 0]
        alpha = calc_alpha_asym(tt, tf, n_s, p[0], n_u, p[1])
        p_ans[0] = p[0] - (1 - 2 * (2 * p[1] - 1) / (
                2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1)))) ** n_s * \
                   (1 - lambda_u / (alpha * tt * p[0])) ** (n_u - 1)
        p_ans[1] = p[1] - (
                1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_s * (p[1] - 2 ** K_s * (1 - p[1]) ** (K_s + 1)))) ** (
                           n_s - 1) * \
                   (1 - lambda_u / (alpha * tt * p[0])) ** n_u
        return p_ans

    p_us = fsolve(psf, [0.9, 0.9], args=(n1, n2, W_2, K_2, lambda1))
    p_su = fsolve(psf, [0.9, 0.9], args=(n2, n1, W_1, K_1, lambda2))
    err1 = np.sqrt(np.sum(np.array(psf(p_us, n2, n1, W_2, K_2, lambda1)) ** 2))
    err2 = np.sqrt(np.sum(np.array(psf(p_su, n1, n2, W_1, K_1, lambda2)) ** 2))
    us = su = True
    if err1 > 1e-5:
        us = False
    if err2 > 1e-5:
        su = False
    p_su = p_su[::-1]   # SMY: reverse p_su so that p_su[0] is group1, p_su[1] is group2
    return p_us, p_su, us, su


def calc_sat_p_fsolve(n, W, K):
    def psf(p):
        p_ans = [0]
        p_ans[0] = p[0] - (1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W * (p[0] - 2 ** K * (1 - p[0]) ** (K + 1)))) ** (
                n - 1)
        return p_ans

    p_s = fsolve(psf, [0.4])
    sat = True
    if not np.isclose(psf(p_s), [0.0]):
        sat = False
    return p_s[0], sat


def calc_as_p_fsolve(n1: int, n2: int, W_1: int, K_1: int, W_2: int, K_2: int):
    def psf(p, n_1, n_2, W_1, K_1, W_2, K_2):
        p_ans = [0, 0]
        p_ans[0] = p[0] - (
                1 - 2 * (2 * p[0] - 1) / (2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1)))) ** (
                           n_1 - 1) * \
                   (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1)))) ** n_2
        p_ans[1] = p[1] - (1 - 2 * (2 * p[0] - 1) / (
                2 * p[0] - 1 + W_1 * (p[0] - 2 ** K_1 * (1 - p[0]) ** (K_1 + 1)))) ** n_1 * \
                   (1 - 2 * (2 * p[1] - 1) / (2 * p[1] - 1 + W_2 * (p[1] - 2 ** K_2 * (1 - p[1]) ** (K_2 + 1)))) ** (
                           n_2 - 1)
        return p_ans

    p_as = fsolve(psf, [0.4, 0.4], args=(n1, n2, W_1, K_1, W_2, K_2))
    err = np.sqrt(np.sum(np.array(psf(p_as, n1, n2, W_1, K_1, W_2, K_2)) ** 2))
    ss = True
    if err > 1e-5:
        ss = False
    return p_as, ss


def calc_access_delay_u(p: float, alpha: float, tt: float, tf: float, W: int, K: int, ilambda: float):
    """calculate queuing delay, access delay

    Params:
    ilambda: input rate per node per slot
    tt: tau_T
    tf: tau_F
    W: length of the initial contend window
    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    tt += 1
    tf += 1
    alpha /= 1 - ilambda * (1 + tf / tt * (1 - p) / p)
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) +
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (
                    p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K + 1)]
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (
            3 * alpha ** 2) for i in range(K + 1)]

    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret

    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1 - p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in
                 range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (
                    1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1
    # according to Geo/G/1's theoretical formula
    ED0_2_L = tt ** 2 + (1 + W) * tt + (1 + 3 * W + 2 * W ** 2) / 6
    queuing_delay = ilambda / tt * (ED0_2 - ED0_1) / (2 * (1 - ilambda / tt * ED0_1))
    access_delay = ED0_1
    # ED01_L = tt + (1 + W) / 2
    # print(ED0_1, ED01_L)
    total_delay = queuing_delay + access_delay
    return queuing_delay, access_delay, ED0_2


def calc_access_delay_s(p: float, alpha: float, tt: float, tf: float, W: int, K: int, ilambda: float, mu_S: float = 0):
    """calculate queuing delay, access delay

    Returns:
        Tuple[float, float]: queuing delay, access delay
    """
    alpha /= 1 - mu_S * (1 + tf / tt * (1 - p) / p)
    ED0_1 = tt + (1 - p) / p * tf + 1 / alpha * (1 / (2 * p) +
                                                 W / 2 * (1 / (2 * p - 1) - 2 ** K * (1 - p) ** (K + 1) / (
                    p * (2 * p - 1))))
    G1Y = [1 / (2 * alpha) * (W * 2 ** i + 1) for i in range(K + 1)]
    # print("G1Y", G1Y)
    G2Y = [1 / (3 * alpha ** 2) * (W ** 2) * 2 ** (2 * i) + (1 - alpha) / alpha ** 2 * W * 2 ** i + (2 - 3 * alpha) / (
            3 * alpha ** 2) for i in range(K + 1)]

    # print("G2Y", G2Y)
    def sum_iK(i):
        ret = 0
        for j in range(i, K):
            ret += (1 - p) ** j * (p * tt + (1 - p) * tf + G1Y[j])
        return ret

    # calculate G''D0(1)
    A = sum([(1 - p) ** i * G2Y[i] for i in range(K)])
    B = sum([(1 - p) ** i * G1Y[i] for i in range(K)])
    C = (1 - p) ** K / p * G2Y[K]
    D = (1 - p) ** K / p * G1Y[K]
    G2D01 = A + C + 2 * (p * tt + (1 - p) * tf) * (B + D) + 2 * \
            sum([(tf + G1Y[i]) * (sum_iK(i + 1) + (1 - p) ** K / p * (p * tt + (1 - p) * tf + G1Y[K])) for i in
                 range(K)]) \
            + 2 * (1 - p) ** (K + 1) / p ** 2 * (tf + G1Y[K]) * (p * tt + (1 - p) * tf + G1Y[K]) + tt * (tt - 1) + (
                    1 - p) / p * (tf ** 2 - tf)
    ED0_2 = G2D01 + ED0_1
    # print(G2D01, ED0_1)
    queueing_delay = mu_S / tt * (ED0_2 - ED0_1) / (2 * (1 - mu_S / tt * ED0_1) + 1e-12)
    access_delay = ED0_1
    # print("queueing delay", queueing_delay)
    return queueing_delay, access_delay, ED0_2
