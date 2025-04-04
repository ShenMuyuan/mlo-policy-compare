import numpy as np
import pandas as pd
from hol_ana_optim import get_double_link_analysis
from congestion_aware import congestion_aware_beta
from condition_aware import condition_aware_beta


if __name__ == "__main__":
    tau_df = pd.read_csv('tau_values.csv')
    n_mld = 5
    n_sld_link1 = 5
    n_sld_link2 = 5
    lambda_mld = 0.002
    lambda_sld_link1 = 0.001
    lambda_sld_link2 = 0.004
    # MCS: 0 - 11
    mcs1 = 6
    mcs2 = 6
    # BW: 20, 40, 80
    bw1 = 20
    bw2 = 20
    # Payload: 500, 1000, 1500, 2000
    payload = 1500
    W_mld_link1_list = [16]
    W_mld_link2_list = [16]
    beta_list = np.clip(np.arange(0, 1.001, 0.001), None, 1)
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
        r_1 = res_df_link1.get('data_bps').tolist()[0]
    else:
        print(f'For link 1, mcs == {mcs1} and bw == {bw1} and payload == {payload} not found in CSV')
        assert False

    res_df_link2 = tau_df.query(f'mcs == {mcs2} and bw == {bw2} and payload == {payload}')
    if not res_df_link2.empty:
        tt_link2, tf_link2 = res_df_link2.get('tau_t_slots').tolist()[0], \
            res_df_link2.get('tau_f_slots').tolist()[0]
        r_2 = res_df_link2.get('data_bps').tolist()[0]
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
    lowest_jitter_ele = df.loc[df['secRawMo'].idxmin()]

    cong_beta = congestion_aware_beta(n_mld, n_sld_link1, n_sld_link2, 
                                  beta_list, tt_link1, tt_link2, tf_link1, tf_link2, 
                                  lambda_mld, lambda_sld_link1, lambda_sld_link2)

    cond_beta = condition_aware_beta(n_mld, n_sld_link1, n_sld_link2, 
                                    beta_list, r_1, r_2, tt_link1, tt_link2, tf_link1, tf_link2, 
                                    lambda_mld, lambda_sld_link1, lambda_sld_link2)
    
    # print(f"Model min. mean E2E delay: beta={lowest_mean_e2e_ele['beta']}, w1={lowest_mean_e2e_ele['w1']}, w2={lowest_mean_e2e_ele['w2']}, meanE2e={lowest_mean_e2e_ele['meanE2e']}, secRawMo={lowest_mean_e2e_ele['secRawMo']}")
    # print(f"Model min. jitter: beta={lowest_jitter_ele['beta']}, w1={lowest_jitter_ele['w1']}, w2={lowest_jitter_ele['w2']}, meanE2e={lowest_jitter_ele['meanE2e']}, secRawMo={lowest_jitter_ele['secRawMo']}")

    print()
    print(f"Congestion aware (CGA): beta={cong_beta}")
    print(f"Condition aware (CDA): beta={cond_beta}")
    print(f"Model minimize mean E2E delay: beta={lowest_mean_e2e_ele['beta']}")
    print(f"Model minimize jitter: beta={lowest_jitter_ele['beta']}")
