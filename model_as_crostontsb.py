# MODEL TIME SERIES WITH CROSTONTSB MODELLING

import os
import numpy as np
import pandas as pd
import sys

def Croston_TSB(ts, extra_periods=1, alpha=0.4, beta=0.4):
    """
    Apply the Croston-Tesla-Slack-Billing (CrostonTSB) model to forecast intermittent demand.
    """
    ts = np.array(ts)
    n = len(ts)
    ts = np.append(ts, [np.nan] * extra_periods)
    a, p, f = np.full((3, n + extra_periods), np.nan)

    first_occurrence = np.argmax(ts[:n] > 0)
    a[0] = ts[first_occurrence]
    p[0] = 1 / (1 + first_occurrence)
    f[0] = p[0] * a[0]

    for t in range(0, n):
        if ts[t] > 0:
            a[t + 1] = alpha * ts[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * 1 + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    a[n + 1:] = a[n]
    p[n + 1:] = p[n]
    f[n + 1:] = f[n]

    return pd.DataFrame({"Demand": ts, "Forecast": f, "Period": p, "Level": a, "Error": ts - f})


def model_as_crostontsb_multivariate(data):
    """
    Apply CrostonTSB model to multivariate data grouped by unique identifiers.
    data = DataFrame with columns 'eid', 't', 'atc_code', and 'orig_value'.
    """
    ind_list = data['eid'].unique().tolist()
    drug_list = data['atc_code'].unique().tolist()

    results = []
    for drug in drug_list:
        sub_data = data[data['atc_code'] == drug]
        temp_results = []

        for eid in ind_list:
            sub_ind_data = sub_data[sub_data['eid'] == eid]
            if sub_ind_data.empty:
                continue
            subdf = sub_ind_data[['eid', 't', 'atc_code']].reset_index(drop=True)
            ts = sub_ind_data.set_index('t')['orig_value']
            croston_rep = Croston_TSB(ts)['Forecast']
            test = pd.DataFrame(croston_rep, columns=["crostontsb"])
            subdf = subdf.join(test)
            temp_results.append(subdf)

        if temp_results:
            results.append(pd.concat(temp_results))

    return pd.concat(results) if results else pd.DataFrame()


# MAIN

def main():
    data_path = '/data_path/'
    file = 'month_N06A_time_series.txt'

    try:
        data = pd.read_table(os.path.join(data_path, file))
    except FileNotFoundError:
        sys.exit("Error: Data file not found. Please check the file path.")

    required_columns = {'eid', 't', 'atc_code', 'orig_value'}
    if not required_columns.issubset(data.columns):
        sys.exit("Error: Missing required columns in the input data.")

    data = data[['eid', 't', 'atc_code', 'orig_value']]
    results = model_as_crostontsb_multivariate(data)

    final_data = pd.merge(data, results, on=['eid', 't', 'atc_code'], how='left')
    output_file = os.path.join(data_path, 'month_N06A_time_series_with_crostonTSB.txt')
    final_data.to_csv(output_file, header=True, index=False, sep='\t')

if __name__ == "__main__":
    main()
