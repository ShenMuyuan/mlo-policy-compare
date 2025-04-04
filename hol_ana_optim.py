import numpy as np
import pandas as pd
from hol_utils import (
    calc_mu_S,
    calc_alpha_sym,
    calc_alpha_asym,
    calc_unsat_p_fsolve,
    calc_sat_p_fsolve,
    calc_au_p_fsolve,
    calc_ps_p_fsolve,
    calc_as_p_fsolve,
    calc_access_delay_s,
    calc_access_delay_u,
    calc_conf
)


# lambda: arrival rate per node, per slot
# tt and tf: holding time in slots
def get_single_link_analysis_one_group(n, lamb, W, K, tt, tf):
    lamb = lamb * tt
    p_unsat, flag_unsat = calc_unsat_p_fsolve(n, lamb, tt, tf)
    if flag_unsat:
        alpha_u = calc_alpha_sym(tt, tf, n, p_unsat)
        queueing_delay, access_delay, access_delay_2nd_raw_moment = calc_access_delay_u(p_unsat, alpha_u, tt, tf, W, K, lamb)
        queueing_delay = np.inf if queueing_delay < 0 or queueing_delay > 1e9 else queueing_delay
        state = "U"
        thpt = n * lamb
        return state, p_unsat, thpt, queueing_delay, access_delay, access_delay_2nd_raw_moment
    else:
        p_sat, _ = calc_sat_p_fsolve(n, W, K)
        alpha_s = calc_alpha_sym(tt, tf, n, p_sat)
        pi_ts = calc_mu_S(p_sat, tt, tf, W, K)
        queueing_delay, access_delay, access_delay_2nd_raw_moment = calc_access_delay_s(p_sat, alpha_s, tt, tf, W, K, lamb, pi_ts)
        queueing_delay = np.inf if queueing_delay < 0 or queueing_delay > 1e9 else queueing_delay
        state = "S"
        thpt = min(lamb, pi_ts) * n
        return state, p_sat, thpt, queueing_delay, access_delay, access_delay_2nd_raw_moment


# lambda: arrival rate per node, per slot
# tt and tf: holding time in slots
def get_single_link_analysis_two_group(
        n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf
):
    if n1 == 0 and n2 == 0:
        return '__', -1, -1, 0, 0, -1, -1, -1, -1
    elif n1 == 0:
        state2, p2, thpt2, qd2, ad2, ad2_2nd_raw_moment = get_single_link_analysis_one_group(n2, lambda2, W_2, K_2, tt, tf)
        return (('_' + state2),
                -1, p2,
                0, thpt2 / tt * 1500 * 8 / 9,
                -1, qd2 * 0.009,
                -1, ad2 * 0.009,
                -1, ad2_2nd_raw_moment * (0.009 ** 2),
                # Second central moment (variance) is calculated using Var[X] = E[X^2] - (E[X])^2
                -1, (ad2_2nd_raw_moment - (ad2 ** 2)) * (0.009 ** 2))
    elif n2 == 0:
        state1, p1, thpt1, qd1, ad1, ad1_2nd_raw_moment = get_single_link_analysis_one_group(n1, lambda1, W_1, K_1, tt, tf)
        return ((state1 + "_"),
                p1, -1,
                thpt1 / tt * 1500 * 8 / 9, 0,
                qd1 * 0.009, -1,
                ad1 * 0.009, -1,
                ad1_2nd_raw_moment * (0.009 ** 2), -1,
                # Second central moment (variance) is calculated using Var[X] = E[X^2] - (E[X])^2
                (ad1_2nd_raw_moment - (ad1 ** 2)) * (0.009 ** 2), -1)
    # print("params for get_single_link_analysis_two_group:", n1, n2, lambda1, lambda2, W_1, W_2, K_1, K_2, tt, tf)
    # Conversion
    lambda1 = lambda1 * tt
    lambda2 = lambda2 * tt
    p_au, flag_au = calc_au_p_fsolve(n1, lambda1, n2, lambda2, tt, tf)
    flag_not_au = False
    if flag_au:
        state = "UU"
        succ_prob_1 = p_au[0]
        succ_prob_2 = p_au[1]
        alpha_au = calc_alpha_asym(tt, tf, n1, p_au[0], n2, p_au[1])
        queuing_delay_1, access_delay_1, access_delay_1_2nd_raw_moment = calc_access_delay_u(
            p_au[0], alpha_au, tt, tf, W_1, K_1, lambda1
        )
        queuing_delay_2, access_delay_2, access_delay_2_2nd_raw_moment = calc_access_delay_u(
            p_au[1], alpha_au, tt, tf, W_2, K_2, lambda2
        )
        throughput_1 = lambda1 * n1
        throughput_2 = lambda2 * n2
        if queuing_delay_1 < 0 or queuing_delay_2 < 0:
            flag_not_au = True
    if not flag_au or flag_not_au:
        p_as, _ = calc_as_p_fsolve(n1, n2, W_1, K_1, W_2, K_2)
        alpha_as = calc_alpha_asym(tt, tf, n1, p_as[0], n2, p_as[1])
        alpha_as_1 = 1 / (1 + tf - tf * p_as[0] - (tt - tf) * p_as[0] * np.log(p_as[0]))
        alpha_as_2 = 1 / (1 + tf - tf * p_as[1] - (tt - tf) * p_as[1] * np.log(p_as[1]))
        cf_as = calc_conf(p_as[0], p_as[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, alpha_as, 3)
        # pi_ts_1 = calc_mu_S(p_as[0], tt, tf, W_1, K_1, alpha_as)
        # pi_ts_2 = calc_mu_S(p_as[1], tt, tf, W_2, K_2, alpha_as)
        pi_ts_1 = calc_mu_S(p_as[0], tt, tf, W_1, K_1, alpha_as_1)
        pi_ts_2 = calc_mu_S(p_as[1], tt, tf, W_2, K_2, alpha_as_2)
        p_us, p_su, _, _ = calc_ps_p_fsolve(
            n1, lambda1, n2, lambda2, W_1, K_1, W_2, K_2, tt, tf
        )
        alpha_us = calc_alpha_asym(tt, tf, n1, p_us[0], n2, p_us[1])
        alpha_us_1 = 1 / (1 + tf - tf * p_us[0] - (tt - tf) * p_us[0] * np.log(p_us[0]))
        alpha_us_2 = 1 / (1 + tf - tf * p_us[1] - (tt - tf) * p_us[1] * np.log(p_us[1]))
        cf_us = calc_conf(
            p_us[0], p_us[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, alpha_us, 1
        )
        alpha_su = calc_alpha_asym(tt, tf, n1, p_su[0], n2, p_su[1])
        alpha_su_1 = 1 / (1 + tf - tf * p_su[0] - (tt - tf) * p_su[0] * np.log(p_su[0]))
        alpha_su_2 = 1 / (1 + tf - tf * p_su[1] - (tt - tf) * p_su[1] * np.log(p_su[1]))
        cf_su = calc_conf(
            p_su[0], p_su[1], lambda1, lambda2, W_1, K_1, W_2, K_2, tt, tf, alpha_su, 2
        )
        cf_list = [cf_us, cf_su, cf_as]
        # print("cf_list:", cf_list)
        assert np.max(cf_list) != 0
        best_idx = np.argmax(cf_list)
        if best_idx == 0:  # US
            state = "US"
            succ_prob_1 = p_us[0]
            succ_prob_2 = p_us[1]
            pi_ts_us2 = calc_mu_S(p_us[1], tt, tf, W_2, K_2)
            queuing_delay_1, access_delay_1, access_delay_1_2nd_raw_moment = calc_access_delay_u(
                # p_us[0], alpha_us, tt, tf, W_1, K_1, lambda1
                p_us[0], alpha_us_1, tt, tf, W_1, K_1, lambda1
            )
            queuing_delay_2, access_delay_2, access_delay_2_2nd_raw_moment = calc_access_delay_s(
                # p_us[1], alpha_us, tt, tf, W_2, K_2, lambda2, pi_ts_us2
                p_us[1], alpha_us_2, tt, tf, W_2, K_2, lambda2, pi_ts_us2
            )
            queuing_delay_1 = np.inf if queuing_delay_1 < 0 or queuing_delay_1 > 1e9 else queuing_delay_1
            queuing_delay_2 = np.inf if queuing_delay_2 < 0 or queuing_delay_2 > 1e9 else queuing_delay_2
            throughput_1 = lambda1 * n1
            throughput_2 = min(lambda2, pi_ts_us2) * n2
        elif best_idx == 1:  # SU
            state = "SU"
            succ_prob_1 = p_su[0]
            succ_prob_2 = p_su[1]
            pi_ts_su1 = calc_mu_S(p_su[0], tt, tf, W_1, K_1)
            queuing_delay_1, access_delay_1, access_delay_1_2nd_raw_moment = calc_access_delay_s(
                # p_su[0], alpha_su, tt, tf, W_1, K_1, lambda1, pi_ts_su1
                p_su[0], alpha_su_1, tt, tf, W_1, K_1, lambda1, pi_ts_su1
            )
            queuing_delay_2, access_delay_2, access_delay_2_2nd_raw_moment = calc_access_delay_u(
                # p_su[1], alpha_su, tt, tf, W_2, K_2, lambda2
                p_su[1], alpha_su_2, tt, tf, W_2, K_2, lambda2
            )
            queuing_delay_1 = np.inf if queuing_delay_1 < 0 or queuing_delay_1 > 1e9 else queuing_delay_1
            queuing_delay_2 = np.inf if queuing_delay_2 < 0 or queuing_delay_2 > 1e9 else queuing_delay_2
            throughput_1 = min(lambda1, pi_ts_su1) * n1
            throughput_2 = lambda2 * n2
        elif best_idx == 2:  # SS
            state = "SS"
            succ_prob_1 = p_as[0]
            succ_prob_2 = p_as[1]
            queuing_delay_1, access_delay_1, access_delay_1_2nd_raw_moment = calc_access_delay_s(
                # p_as[0], alpha_as, tt, tf, W_1, K_1, lambda1, pi_ts_1
                p_as[0], alpha_as_1, tt, tf, W_1, K_1, lambda1, pi_ts_1
            )
            queuing_delay_2, access_delay_2, access_delay_2_2nd_raw_moment = calc_access_delay_s(
                # p_as[1], alpha_as, tt, tf, W_2, K_2, lambda2, pi_ts_2
                p_as[1], alpha_as_2, tt, tf, W_2, K_2, lambda2, pi_ts_2
            )
            queuing_delay_1 = np.inf if queuing_delay_1 < 0 or queuing_delay_1 > 1e9 else queuing_delay_1
            queuing_delay_2 = np.inf if queuing_delay_2 < 0 or queuing_delay_2 > 1e9 else queuing_delay_2
            throughput_1 = min(lambda1, pi_ts_1) * n1
            throughput_2 = min(lambda2, pi_ts_2) * n2

    # print("state is", state)
    # print("p1 is", succ_prob_1)
    # print("p2 is", succ_prob_2)

    # Including conversion
    return (
        state,
        succ_prob_1,
        succ_prob_2,
        throughput_1 / tt * 1500 * 8 / 9,
        throughput_2 / tt * 1500 * 8 / 9,
        queuing_delay_1 * 0.009,
        queuing_delay_2 * 0.009,
        access_delay_1 * 0.009,
        access_delay_2 * 0.009,
        # Second raw moment of access delay
        access_delay_1_2nd_raw_moment * (0.009 ** 2),
        access_delay_2_2nd_raw_moment * (0.009 ** 2),
        # Second central moment (variance) is calculated using Var[X] = E[X^2] - (E[X])^2
        (access_delay_1_2nd_raw_moment - (access_delay_1 ** 2)) * (0.009 ** 2),
        (access_delay_2_2nd_raw_moment - (access_delay_2 ** 2)) * (0.009 ** 2)
    )


# lambda: arrival rate per node, per slot
# tt and tf: holding time in slots
def get_double_link_analysis(
        n_mld,
        n_sld_link1,
        n_sld_link2,
        lambda_mld,
        beta,
        lambda_sld_link1,
        lambda_sld_link2,
        W_mld_link1,
        W_mld_link2,
        W_sld_link1,
        W_sld_link2,
        K_mld_link1,
        K_mld_link2,
        K_sld_link1,
        K_sld_link2,
        tt_link1,
        tt_link2,
        tf_link1,
        tf_link2,
):
    lambda_mld_link1 = lambda_mld * beta
    lambda_mld_link2 = lambda_mld * (1 - beta)
    # Analysis for Link 1
    (
        l1_state,
        l1_succ_prob_mld,
        l1_succ_prob_sld,
        l1_throughput_mld,
        l1_throughput_sld,
        l1_queuing_delay_mld,
        l1_queuing_delay_sld,
        l1_access_delay_mld,
        l1_access_delay_sld,
        l1_access_delay_mld_2nd_raw_moment,
        l1_access_delay_sld_2nd_raw_moment,
        l1_access_delay_mld_2nd_central_moment,
        l1_access_delay_sld_2nd_central_moment,
    ) = get_single_link_analysis_two_group(
        n_mld,
        n_sld_link1,
        lambda_mld_link1,
        lambda_sld_link1,
        W_mld_link1,
        W_sld_link1,
        K_mld_link1,
        K_sld_link1,
        tt_link1,
        tf_link1,
    )
    # Analysis for Link 2
    (
        l2_state,
        l2_succ_prob_mld,
        l2_succ_prob_sld,
        l2_throughput_mld,
        l2_throughput_sld,
        l2_queuing_delay_mld,
        l2_queuing_delay_sld,
        l2_access_delay_mld,
        l2_access_delay_sld,
        l2_access_delay_mld_2nd_raw_moment,
        l2_access_delay_sld_2nd_raw_moment,
        l2_access_delay_mld_2nd_central_moment,
        l2_access_delay_sld_2nd_central_moment,
    ) = get_single_link_analysis_two_group(
        n_mld,
        n_sld_link2,
        lambda_mld_link2,
        lambda_sld_link2,
        W_mld_link2,
        W_sld_link2,
        K_mld_link2,
        K_sld_link2,
        tt_link2,
        tf_link2,
    )

    return (
        l1_state,
        l1_succ_prob_mld,
        l1_succ_prob_sld,
        l1_throughput_mld,
        l1_throughput_sld,
        l1_queuing_delay_mld,
        l1_queuing_delay_sld,
        l1_access_delay_mld,
        l1_access_delay_sld,
        l1_access_delay_mld_2nd_raw_moment,
        l1_access_delay_sld_2nd_raw_moment,
        l1_access_delay_mld_2nd_central_moment,
        l1_access_delay_sld_2nd_central_moment,
        l2_state,
        l2_succ_prob_mld,
        l2_succ_prob_sld,
        l2_throughput_mld,
        l2_throughput_sld,
        l2_queuing_delay_mld,
        l2_queuing_delay_sld,
        l2_access_delay_mld,
        l2_access_delay_sld,
        l2_access_delay_mld_2nd_raw_moment,
        l2_access_delay_sld_2nd_raw_moment,
        l2_access_delay_mld_2nd_central_moment,
        l2_access_delay_sld_2nd_central_moment,
    )


if __name__ == "__main__":
    tau_df = pd.read_csv('tau_values_copy.csv')
    n_mld = 5
    # n_sld_link1 = 8
    # n_sld_link2 = 16
    # n_sld_link1 = 6
    # n_sld_link2 = 18
    n_sld_link1 = 5
    n_sld_link2 = 5
    lambda_mld = 0.002
    lambda_sld_link1 = 0.001
    lambda_sld_link2 = 0.001
    # MCS: 0 - 11
    mcs1 = 6
    mcs2 = 10
    # BW: 20, 40, 80
    bw1 = 20
    bw2 = 40
    # Payload: 1000, 1500, 2000, ..., 100000
    payload = 1500
    # W_mld_link1_list = [2, 4, 8, 16]
    # W_mld_link2_list = [2, 4, 8, 16]
    W_mld_link1_list = [16]
    W_mld_link2_list = [16]
    beta_list = np.clip(np.arange(0.001, 1.001, 0.001), None, 1)
    W_sld_link1 = 16
    W_sld_link2 = 16
    K_mld_link1 = 6
    K_mld_link2 = 6
    K_sld_link1 = 6
    K_sld_link2 = 6

    res_df_link1 = tau_df.query(f'mcs == {mcs1} and bw == {bw1} and payload == {payload}')
    if not res_df_link1.empty:
        tt_link1, tf_link1 = res_df_link1.get('tau_t_slots').tolist()[0], \
            res_df_link1.get('tau_f_slots').tolist()[0]
    else:
        print(f'For link 1, mcs == {mcs1} and bw == {bw1} and payload == {payload} not found in CSV')
        assert False

    res_df_link2 = tau_df.query(f'mcs == {mcs2} and bw == {bw2} and payload == {payload}')
    if not res_df_link2.empty:
        tt_link2, tf_link2 = res_df_link2.get('tau_t_slots').tolist()[0], \
            res_df_link2.get('tau_f_slots').tolist()[0]
    else:
        print(f'For link 2, mcs == {mcs2} and bw == {bw2} and payload == {payload} not found in CSV')
        assert False

    cols = ['beta', 'w1', 'w2', 'meanE2e', 'secRawMo']
    df = pd.DataFrame(columns=cols)

    for beta in beta_list:
        for W_mld_link1 in W_mld_link1_list:
            for W_mld_link2 in W_mld_link2_list:
                (
                    l1_state,
                    mldSuccPrLink1,
                    sldLink1SuccPr,
                    mldThptLink1,
                    sldLink1Thpt,
                    mldMeanQueDelayLink1,
                    sldMeanQueDelayLink1,
                    mldMeanAccDelayLink1,
                    sldMeanAccDelayLink1,
                    mld2ndRawMomentAccDelayLink1,
                    sld2ndRawMomentAccDelayLink1,
                    mld2ndCentralMomentAccDelayLink1,
                    sld2ndCentralMomentAccDelayLink1,
                    l2_state,
                    mldSuccPrLink2,
                    sldLink2SuccPr,
                    mldThptLink2,
                    sldLink2Thpt,
                    mldMeanQueDelayLink2,
                    sldMeanQueDelayLink2,
                    mldMeanAccDelayLink2,
                    sldMeanAccDelayLink2,
                    mld2ndRawMomentAccDelayLink2,
                    sld2ndRawMomentAccDelayLink2,
                    mld2ndCentralMomentAccDelayLink2,
                    sld2ndCentralMomentAccDelayLink2,
                ) = get_double_link_analysis(
                    n_mld,
                    n_sld_link1,
                    n_sld_link2,
                    lambda_mld,
                    beta,
                    lambda_sld_link1,
                    lambda_sld_link2,
                    W_mld_link1,
                    W_mld_link2,
                    W_sld_link1,
                    W_sld_link2,
                    K_mld_link1,
                    K_mld_link2,
                    K_sld_link1,
                    K_sld_link2,
                    tt_link1,
                    tt_link2,
                    tf_link1,
                    tf_link2,
                )

                mldSuccPrTotal = ((mldSuccPrLink1 * mldThptLink1
                                   + mldSuccPrLink2 * mldThptLink2)
                                  / (mldThptLink1 + mldThptLink2)) if (mldThptLink1 + mldThptLink2 != 0) else -1
                mldThptTotal = mldThptLink1 + mldThptLink2
                mldMeanQueDelayTotal = ((mldMeanQueDelayLink1 * mldThptLink1
                                         + mldMeanQueDelayLink2 * mldThptLink2)
                                        / (mldThptLink1 + mldThptLink2)) if (mldThptLink1 + mldThptLink2 != 0) else -1
                mldMeanAccDelayTotal = ((mldMeanAccDelayLink1 * mldThptLink1
                                         + mldMeanAccDelayLink2 * mldThptLink2)
                                        / (mldThptLink1 + mldThptLink2)) if (mldThptLink1 + mldThptLink2 != 0) else -1
                mld2ndRawMomentAccDelayTotal = ((mld2ndRawMomentAccDelayLink1 * mldThptLink1
                                                 + mld2ndRawMomentAccDelayLink2 * mldThptLink2)
                                                / (mldThptLink1 + mldThptLink2)) if (mldThptLink1 + mldThptLink2 != 0) else -1
                mld2ndCentralMomentAccDelayTotal = ((mld2ndCentralMomentAccDelayLink1 * mldThptLink1
                                                     + mld2ndCentralMomentAccDelayLink2 * mldThptLink2)
                                                    / (mldThptLink1 + mldThptLink2)) if (mldThptLink1 + mldThptLink2 != 0) else -1
                mldMeanE2eDelayLink1 = (mldMeanQueDelayLink1 + mldMeanAccDelayLink1) if (mldMeanQueDelayLink1 != -1) else -1
                mldMeanE2eDelayLink2 = (mldMeanQueDelayLink2 + mldMeanAccDelayLink2) if (mldMeanQueDelayLink2 != -1) else -1
                mldMeanE2eDelayTotal = (mldMeanQueDelayTotal + mldMeanAccDelayTotal) if (mldMeanQueDelayTotal != -1) else -1
                sldMeanE2eDelayLink1 = (sldMeanQueDelayLink1 + sldMeanAccDelayLink1) if (sldMeanQueDelayLink1 != -1) else -1
                sldMeanE2eDelayLink2 = (sldMeanQueDelayLink2 + sldMeanAccDelayLink2) if (sldMeanQueDelayLink2 != -1) else -1
                totalSuccPr = ((mldSuccPrTotal * mldThptTotal
                                + sldLink1SuccPr * sldLink1Thpt
                                + sldLink2SuccPr * sldLink2Thpt)
                               / (mldThptTotal + sldLink1Thpt + sldLink2Thpt)) if (
                        mldThptTotal + sldLink1Thpt + sldLink2Thpt != 0) else -1
                totalThpt = mldThptTotal + sldLink1Thpt + sldLink2Thpt
                totalMeanQueDelay = ((mldMeanQueDelayTotal * mldThptTotal
                                      + sldMeanQueDelayLink1 * sldLink1Thpt
                                      + sldMeanQueDelayLink2 * sldLink2Thpt)
                                     / (mldThptTotal + sldLink1Thpt + sldLink2Thpt)) if (
                        mldThptTotal + sldLink1Thpt + sldLink2Thpt != 0) else -1
                totalMeanAccDelay = ((mldMeanAccDelayTotal * mldThptTotal
                                      + sldMeanAccDelayLink1 * sldLink1Thpt
                                      + sldMeanAccDelayLink2 * sldLink2Thpt)
                                     / (mldThptTotal + sldLink1Thpt + sldLink2Thpt)) if (
                        mldThptTotal + sldLink1Thpt + sldLink2Thpt != 0) else -1
                totalMeanE2eDelay = ((mldMeanE2eDelayTotal * mldThptTotal
                                      + sldMeanE2eDelayLink1 * sldLink1Thpt
                                      + sldMeanE2eDelayLink2 * sldLink2Thpt)
                                     / (mldThptTotal + sldLink1Thpt + sldLink2Thpt)) if (
                        mldThptTotal + sldLink1Thpt + sldLink2Thpt != 0) else -1

                new_row = pd.Series({'beta': beta, 'w1': W_mld_link1, 'w2': W_mld_link2, 'meanE2e': mldMeanE2eDelayTotal, 'secRawMo': mld2ndRawMomentAccDelayTotal})
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    lowest_mean_e2e_ele = df.loc[df['meanE2e'].idxmin()]
    print(f"Min. mean E2E delay: beta={lowest_mean_e2e_ele['beta']}, w1={lowest_mean_e2e_ele['w1']}, w2={lowest_mean_e2e_ele['w2']}, meanE2e={lowest_mean_e2e_ele['meanE2e']}, secRawMo={lowest_mean_e2e_ele['secRawMo']}")

    lowest_jitter_ele = df.loc[df['secRawMo'].idxmin()]
    print(f"Min. jitter: beta={lowest_jitter_ele['beta']}, w1={lowest_jitter_ele['w1']}, w2={lowest_jitter_ele['w2']}, meanE2e={lowest_jitter_ele['meanE2e']}, secRawMo={lowest_jitter_ele['secRawMo']}")
