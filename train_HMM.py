# Training a Hidden Markov Model (HMM) with n states
# This script trains an HMM on 75% of the dataset and fits it to the entire dataset.

import os
import sys
import getopt
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

print("Import of packages successful")


# Read and preprocess the data
def read_the_data(representation):
    """
    Reads and processes time series data based on the chosen representation.
    
    Args:
        representation (str): The representation type ('croston', 'crostontsb', 'orig_value').
    
    Returns:
        pd.DataFrame: Processed dataset.
    """
    # Define file paths
    ts_data_path = 'path_to_time_series_data/'
    crostontsb_subdir_path = 'month_N06A_time_series_16052022_with_crostonTSB_subsets/'

    if representation == 'crostontsb':
        data = pd.DataFrame()
        folder_path = os.path.join(ts_data_path, crostontsb_subdir_path)
        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            print(f"Folder not found: {folder_path}")
            sys.exit(1)
        for f in files:
            file_path = os.path.join(folder_path, f)
            temp_df = pd.read_table(file_path)
            temp_df = temp_df[['eid', 't', 'atc_code', representation]]
            data = pd.concat([data, temp_df], ignore_index=True)
          
    return data


def apply_additional_exclusion_criteria(data):
    """
    Applies additional exclusion criteria to the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset.
    
    Returns:
        pd.DataFrame: Dataset after applying exclusion criteria.
    """
    # Define exclusion files
    individuals_to_drop_path = 'individuals_to_drop.txt'
    drugs_to_drop_path = 'N06A_drugs_to_drop.txt'

    try:
        ind_to_drop = pd.read_table(individuals_to_drop_path)
    except FileNotFoundError:
        print(f"File not found: {individuals_to_drop_path}")
        sys.exit(1)

    data = data[~data['id'].isin(np.unique(ind_to_drop))]
    number_of_ind = len(np.unique(data['id']))
    print(f"The number of individuals in this cohort is {number_of_ind}")

    try:
        final_drugs_to_drop = pd.read_table(drugs_to_drop_path)
    except FileNotFoundError:
        print(f"File not found: {drugs_to_drop_path}")
        sys.exit(1)

    data = data[~data['atc_code'].isin(np.unique(final_drugs_to_drop['atc_to_drop']))]
    number_of_drugs = len(np.unique(data['atc_code']))
    print(f"The number of drugs in this cohort is {number_of_drugs}")

    return data


def calculate_length_time_series(data):
    """
    Calculates the length of time series for each unique individual.
    
    Args:
        data (pd.DataFrame): Input dataset.
    
    Returns:
        np.ndarray: Array of time series lengths.
    """
    time_series_lengths = data.groupby('id')['t'].max().values
    return time_series_lengths
  

def format_data_for_hmm(data, chunk_size=5000):
    # Initialize empty arrays to store results
    data_as_mvarray = []
    # Calculate the total number of EIDs
    num_eids = len(data['eid'].unique())
    # Iterate through chunks of data
    for start in range(0, num_eids, chunk_size):
        end = start + chunk_size
        chunk_eids = data['eid'].unique()[start:end]
        # Filter data for the current chunk of EIDs
        chunk_data = data[data['eid'].isin(chunk_eids)]
        # Pivot and process data for the chunk
        df = chunk_data.pivot_table(
            values=representation,
            index=['eid', 't'],
            columns='atc_code'
        )
        #print(df.columns)
        chunk_data_as_mvarray = np.ceil(df.to_numpy())
        # Append the results for the chunk to the arrays
        data_as_mvarray.append(chunk_data_as_mvarray)
    # Concatenate results from all chunks
    final_data_as_mvarray = np.concatenate(data_as_mvarray, axis=0)
    return final_data_as_mvarray


def selection_measures2(model,X,Xl,n):
    log_likelihood = model.score(X,Xl)
    n_sample, n_features = X.shape
    n_states = model.n_components
    n_transition_params = (n_states-1) * n_states
    n_mean_params = n_states * n_features
    n_cov_params = n_states * n_features * (n_features + 1) // 2
    n_parameters = n_transition_params * n_mean_params + n_cov_params
    aic = -2 * log_likelihood + 2 * n_parameters
    bic = -2 * log_likelihood + n_parameters * np.log(n_sample)
    return(aic,bic)


def run_final_hmm(complete_dataset_original, training_dataset, complete_dataset, length_trainig, length_complete, n_coms = 8, idx=1):
	X = training_dataset
	Xl = length_trainig
	Y = complete_dataset
	Yl = length_complete
	model = hmm.PoissonHMM(n_components=int(n_states),n_iter=100,random_state=int(idx))
	model.fit(X,Xl)
	# Get the parameters of the model:
	params = {
		'logprob': model.score(X,Xl),
		'startprob': model.startprob_,
		'lambdas': model.lambdas_,
		'transmat': model.transmat_,
		'monitor': model.monitor_
	}
	# Calculate AIC & BIC:
	aic,bic = selection_measures2(model, X, Xl, int(n_states))
	# Run the model in the entire dataset:
	Z = model.predict(Y,Yl)
	#zdf = pd.DataFrame(data = Z, columns = ['states'])
	Ydf = complete_dataset_original[['eid','t']].drop_duplicates()
	#mergeDf = Ydf.merge(zdf, left_index=True, right_index=True)
	Ydf['states'] = Z
	return(model,Ydf,params, aic, bic)




# Plot a clustered heatmap:
def plot_lambdas(lambdas_array):
	sns.set(font_scale=1.2)
	g = sns.clustermap(lambdas_array, method='average', cmap='coolwarm', row_cluster=True, col_cluster=True,
		xticklabels=col_labels,figsize=(8, 6))
	# Create custom row annotation
	ax1 = g.ax_heatmap
	ax2 = ax1.twinx()
	#ax2.set_yticks(np.arange(n_states) + 0.5)
	#ax2.set_yticklabels(row_labels)
	plt.savefig(results_dir+'hold_out_test_lambdas_nstates_'+str(n_states)+'_idx_'+str(idx)+'.png')





# MAIN #

# Arguments:
n_states = str(sys.argv[1])
representation = str(sys.argv[2])
idx = str(sys.argv[3])
dataset = 'N06A_selected'
# Create the results dir:
results_dir = '/nfs/research/birney/users/heza/ukbb/results/hmm_results/' + dataset + '/month_ts_rep/' + representation + '/poisson_distribution/final_model/'
isExist = os.path.exists(results_dir)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(results_dir)
  print("The new directory has been created!")

# Read the data and apply the additional exclusion criteria to keep only N06A_selected drugs.
data = read_the_data(representation)
df = apply_additional_exclusion_criteria(data)


# Read the training samples:
training_ind_path = '/nfs/research/birney/users/heza/ukbb/results/hmm_results/' + dataset + '/month_ts_rep/' + representation + '/poisson_distribution/hold_out_test_validation/'+'sample_indiv_used_for_hmm_training.txt'
training_ind = pd.read_table(training_ind_path)
# And prepare the data:
training_df = df[df['eid'].isin(np.unique(training_ind['eid']))]
#test_df = df[~df['eid'].isin(np.unique(training_ind['eid']))]
print('the length of the training dataset is '+str(len(np.unique(training_df['eid']))))
#print('the length of the test dataset is '+str(len(np.unique(test_df['eid']))))
# Calculate the length for the two splits:
ltraining = calculate_length_time_series(training_df)
ldf = calculate_length_time_series(df)

# Format the data for the HMM:
forTraining = format_data_for_hmm(training_df,10000)
forComplete = format_data_for_hmm(df,10000)

# Train and fit the model:
mymodel,output, params,aic,bic = run_final_hmm(df, forTraining, forComplete, ltraining, ldf, n_coms = n_states, idx = idx)
summary_output = output['states'].value_counts(normalize=True) * 100

# Save the model:
filename = results_dir +'model_hmm'+n_states+'_'+representation+'_idx_'+idx+'.pkl'
pickle.dump(mymodel,open(filename,'wb'))
print('model saved')

# Save the parameters:
with open(results_dir+'final_params_nstates_'+str(n_states)+'_idx_'+str(idx)+'.pkl','wb') as pickle_file:
	pickle.dump(params,pickle_file)

# Save the outputs:
output.to_csv(results_dir+'final_results_nstates_'+str(n_states)+'_idx_'+str(idx)+'.txt',index=False)
summary_output.to_csv(results_dir+'final_summary_results_nstates_'+str(n_states)+'_idx_'+str(idx)+'.txt',index=False)
# Save AIC and BIC:
aicfile = results_dir+'criterium_file.csv'
with open(aicfile,"a+") as file:
    file.write(str(aic) +','+ str(bic) + ',' + idx + "\n")


# Plot the lambdas
tp = params['lambdas']
col_labels = np.sort(np.unique(df['atc_code']))
plot_lambdas(tp)


print("ALL DONE")




