import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from hmmlearn import hmm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Utility Functions
def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory created: {path}")
    else:
        logging.info(f"Directory already exists: {path}")


# Data Preparation
def read_and_process_data(representation, ts_data_path, crostontsb_subdir_path):
    """Read and preprocess data."""
    data = pd.DataFrame()
    folder_path = os.path.join(ts_data_path, crostontsb_subdir_path)

    if representation == 'crostontsb':
        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            logging.error(f"Folder not found: {folder_path}")
            sys.exit(1)

        for f in files:
            file_path = os.path.join(folder_path, f)
            temp_df = pd.read_table(file_path)
            data = pd.concat([data, temp_df[['eid', 't', 'atc_code', representation]]], ignore_index=True)

    return data


def apply_exclusion_criteria(data, ind_to_drop_path, drugs_to_drop_path):
    """Apply exclusion criteria."""
    try:
        ind_to_drop = pd.read_table(ind_to_drop_path)
        data = data[~data['eid'].isin(np.unique(ind_to_drop))]
        logging.info(f"Number of individuals after exclusion: {len(data['eid'].unique())}")
    except FileNotFoundError:
        logging.warning(f"File not found: {ind_to_drop_path}")

    try:
        drugs_to_drop = pd.read_table(drugs_to_drop_path)
        data = data[~data['atc_code'].isin(np.unique(drugs_to_drop['atc_to_drop']))]
        logging.info(f"Number of drugs after exclusion: {len(data['atc_code'].unique())}")
    except FileNotFoundError:
        logging.warning(f"File not found: {drugs_to_drop_path}")

    return data


# Model Training and Evaluation
def train_hmm(X, Xl, n_states, idx):
    """Train the HMM model."""
    model = hmm.PoissonHMM(n_components=n_states, n_iter=100, random_state=idx)
    model.fit(X, Xl)
    logging.info("HMM training complete.")
    return model


def evaluate_hmm(model, X, Xl, Y, Yl):
    """Evaluate HMM and generate outputs."""
    log_likelihood = model.score(X, Xl)
    states = model.predict(Y, Yl)
    params = {
        'logprob': log_likelihood,
        'startprob': model.startprob_,
        'lambdas': model.lambdas_,
        'transmat': model.transmat_,
    }
    return states, params


def save_model_and_results(model, params, output, summary_output, aic, bic, results_dir, n_states, idx, representation):
    """Save the model, parameters, and results."""
    model_file = os.path.join(results_dir, f'model_hmm{n_states}_{representation}_idx_{idx}.pkl')
    params_file = os.path.join(results_dir, f'final_params_nstates_{n_states}_idx_{idx}.pkl')
    output_file = os.path.join(results_dir, f'final_results_nstates_{n_states}_idx_{idx}.txt')
    summary_file = os.path.join(results_dir, f'final_summary_results_nstates_{n_states}_idx_{idx}.txt')
    aic_file = os.path.join(results_dir, 'criterium_file.csv')

    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    with open(params_file, 'wb') as file:
        pickle.dump(params, file)

    output.to_csv(output_file, index=False)
    summary_output.to_csv(summary_file, index=False)

    with open(aic_file, 'a+') as file:
        file.write(f"{aic},{bic},{idx}\n")

    logging.info("Model and results saved.")


# MAIN Functionality
if __name__ == "__main__":
    # Initialize parameters
    n_states = 8
    representation = 'crostontsb'
    idx = 1
    results_dir = 'results_dir'
    ts_data_path = 'path_to_time_series_data'
    crostontsb_subdir_path = 'month_N06A_time_series_with_crostonTSB_subsets'
    training_ind_path = 'sample_indiv_used_for_hmm_training.txt'
    ind_to_drop_path = 'individuals_to_drop.txt'
    drugs_to_drop_path = 'N06A_drugs_to_drop.txt'

    # Create results directory
    create_directory(results_dir)

    # Read and process data
    data = read_and_process_data(representation, ts_data_path, crostontsb_subdir_path)
    data = apply_exclusion_criteria(data, ind_to_drop_path, drugs_to_drop_path)

    # Prepare training data
    training_ind = pd.read_table(training_ind_path)
    training_data = data[data['eid'].isin(training_ind['eid'])]
    length_training = calculate_length_time_series(training_data)
    length_data = calculate_length_time_series(data)

    # Format data for HMM
    formatted_training = format_data_for_hmm(training_data, chunk_size=10000)
    formatted_data = format_data_for_hmm(data, chunk_size=10000)

    # Train HMM
    model = train_hmm(formatted_training, length_training, n_states, idx)
    states, params = evaluate_hmm(model, formatted_training, length_training, formatted_data, length_data)

    # Save results
    summary_output = pd.DataFrame({'states': states}).value_counts(normalize=True) * 100
    save_model_and_results(model, params, data, summary_output, aic=0, bic=0,  # Placeholder for AIC and BIC
                           results_dir=results_dir, n_states=n_states, idx=idx, representation=representation)
    logging.info("All done.")
