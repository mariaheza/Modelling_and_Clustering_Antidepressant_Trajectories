# MODEL TIME SERIES WITH CROSTONTSB MODELLING
# This script can be used with multivariate or univariate input data:

# bsub -M25000 "python model_as_crostontsb.py 55"

import os
import numpy as np
import pandas as pd
import sys, getopt

# Croston function:
def Croston_TSB(ts,extra_periods=1,alpha=0.4,beta=0.4):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    #level (a), periodicity(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/ (1 + first_occurence)
    f[0] = p[0]*a[0]
	# Create all the t+1 forecasts
    for t in range(0,cols):        
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = beta*(1) + (1-beta)*p[t]         
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]
        f[t+1] = p[t+1]*a[t+1]
    # Future Forecast 
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]            
    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return df

def model_as_crostontsb_multivariate(data):
	ind_list = data.eid.unique().tolist() # each row in the dataframe as element in a list
	drug_list = data.atc_code.unique().tolist()
	#
	LD = []
	for d in drug_list:
		subd = data[data['atc_code'] == d]
		L = []
		for i in ind_list:  # iterate though the elements in the list to process them individually
			print(i)
			subs = subd[subd['eid'] == i]  
			subdf = subs[['eid','t', 'atc_code']]
			subdf = subdf.reset_index(drop=True)
			subs = subs.set_index('t')['orig_value']
			croston_rep = Croston_TSB(subs)['Forecast']
			mov_avg = croston_rep.rolling(7).sum()
			# Note that I am not using the mov average!
			test = pd.DataFrame([croston_rep]).T.rename(columns={"Forecast": "crostontsb"})
			subdf = subdf.join(test)
			L.append(subdf)
		results1 = pd.concat(L)
		#
		LD.append(results1)
	#
	results2 = pd.concat(LD)  # to transform it in a datframe
	return(results2)


######################################
################ MAIN ################
######################################


subset_n = str(sys.argv[1])
time_stamp = 'month'
representation = 'crostontsb'
variables = 'multivariate'
subdir = 'month_ts_rep/'
data_path = '/data_path/'+subdir
file = 'month_N06A_time_series_16052022_subset_'+subset_n+'.txt'

data = pd.read_table(data_path+file)
# PROCESS THE DATA:
# To read as pandas dataframe and convert it to numpy array
data.shape
list(data.columns)
# To drop the columns that we don't need:
if representation=='crostontsb': 
	if variables=='univariate':
		df = data[['eid','t','orig_value']]
	else:
		df = data[['eid','t','atc_code','orig_value']]
else:
	df = data[['eid','t',representation]]




if representation == 'crostontsb':
	if variables == 'univariate':
		df2 = model_as_crostontsb(df)
	else:
		df2 = model_as_crostontsb_multivariate(df)
else:
	df2 = df



# Append to the original dataset and save:
final = pd.merge(data,df2, on=['eid','t','atc_code'])
final.to_csv(data_path+'month_N06A_time_series_16052022_with_crostonTSB_subsets/month_N06A_time_series_with_crostonTSB_16052022_subset_'+subset_n+'.txt', header=True, index=False, sep='\t', mode='a')

print('all done for subset '+subset_n)




