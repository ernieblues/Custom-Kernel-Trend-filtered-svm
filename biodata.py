# Simulated Biological Processes

import csv
import math
import numpy as np
import pandas as pd

from IPython.display import display

##################################################
############### Train_11000 Dataset ##############
##################################################

# random_state number
if not exists('./projects/thesis/results/current/program_data/random_state.csv'):
    pd.DataFrame(data=[random.randrange(1, 100)]).to_csv('./projects/thesis/results/current/program_data/random_state.csv', index=False)
        
random_state = pd.read_csv('./projects/thesis/results/current/program_data/random_state.csv').iloc[0,0]
            
# set data types to string
with open("./projects/thesis/data/train_11000/parsed_class/sweep_11000.csv") as f:
    num_col = len(f.readline().strip().split(',')) # number of columns
    dtypes = {i: np.str for i in range(0, num_col)}

# import dataset
neutral_11000 = pd.read_csv("./projects/thesis/data/train_11000/parsed_class/neutral_11000.csv", index_col=0, dtype=dtypes) # no natural selection
sweep_11000 = pd.read_csv("./projects/thesis/data/train_11000/parsed_class/sweep_11000.csv", index_col=0, dtype=dtypes) # natural selection

# cleanup dataset
neutral_11000.rename_axis(None, inplace=True) # remove index name
sweep_11000.rename_axis(None, inplace=True) # remove index name

neutral_11000.apply(lambda x: pd.to_numeric(x, errors='coerce')) # set data types to float64, rows with string changed to NaN
sweep_11000.apply(lambda x: pd.to_numeric(x, errors='coerce')) # set data types to float64, rows with string changed to NaN

neutral_rows_with_nan = [index for index, row in neutral_11000.iterrows() if row.isnull().any()] # get indices of rows with NaN
sweep_rows_with_nan = [index for index, row in sweep_11000.iterrows() if row.isnull().any()] # get indices of rows with NaN

# NaN: set neutral indices = sweep indices
'''for index in neutral_rows_with_nan:
    if not 'sweep_{}'.format(index[5:]) in sweep_rows_with_nan:
        sweep_rows_with_nan.append('sweep_{}'.format(index[5:]))
for index in sweep_rows_with_nan:
    if not 'neut_{}'.format(index[6:]) in neutral_rows_with_nan:
        neutral_rows_with_nan.append('neut_{}'.format(index[6:]))
'''
# NaN: sort indices
for i in range(len(neutral_rows_with_nan)-1):
    j = i
    while j >= 0 and int(neutral_rows_with_nan[j][5:]) > int(neutral_rows_with_nan[j+1][5:]):
        neutral_rows_with_nan.insert(j, neutral_rows_with_nan[j+1])
        del neutral_rows_with_nan[j+2]
        j -= 1
for i in range(len(sweep_rows_with_nan)-1):
    j = i
    while j >= 0 and int(sweep_rows_with_nan[j][6:]) > int(sweep_rows_with_nan[j+1][6:]):
        sweep_rows_with_nan.insert(j, sweep_rows_with_nan[j+1])
        del sweep_rows_with_nan[j+2]
        j -= 1

# drop rows with NaN
neutral_11000 = neutral_11000.drop(neutral_rows_with_nan)
sweep_11000 = sweep_11000.drop(sweep_rows_with_nan) 

# trim sweep_11000 to 11,000 observations
trim_indices = list(range(11001, sweep_11000.shape[0] + 1))
trim_indices = ['sweep_' + str(i) for i in trim_indices]
sweep_11000 = sweep_11000.drop(trim_indices)

print('Observations removed\nneutral NAN: {}\nsweep NAN: {}\nsweep TRIM: {}\n'.format(neutral_rows_with_nan, sweep_rows_with_nan, trim_indices))

# set data types
neutral_11000 = neutral_11000.astype('float64')
sweep_11000 = sweep_11000.astype('float64')

##################################################
###### Classification Dataset: Train_11000 #######
##################################################

# dataset
neutral_11000_class = neutral_11000.copy()
sweep_11000_class = sweep_11000.copy()

# add class
neutral_11000_class['class'] = int(-1)
sweep_11000_class['class'] = int(1)

# shuffle data
neutral_11000_class = neutral_11000_class.sample(frac=1, random_state=random_state)
sweep_11000_class = sweep_11000_class.sample(frac=1, random_state=random_state)

# combine data
sim_bio_proc_class = pd.concat([neutral_11000_class, sweep_11000_class], axis=0)

# name dataset
sim_bio_proc_class.name = 'Train_11000 Classification'

# display dataset
print(sim_bio_proc_class.name)
print(sim_bio_proc_class.shape, '\n')
display(sim_bio_proc_class.head())
display(sim_bio_proc_class.iloc[sim_bio_proc_class.shape[0]-5:sim_bio_proc_class.shape[0], :])

##################################################
######### Regression Dataset: Train_11000 ########
##################################################

# import regression targets
sweep_11000_target = pd.read_csv("./projects/thesis/data/train_11000/parsed_reg/sweep_parameters.csv", index_col=0) # natural selection

# cleanup data
sweep_11000_target.rename_axis(None, inplace=True) # remove index name
sweep_11000_target = sweep_11000_target.drop(sweep_rows_with_nan + trim_indices) # drop rows with NaN

# combine data
sim_bio_proc_reg = pd.concat((sweep_11000, sweep_11000_target), axis=1)

# shuffle data
sim_bio_proc_reg = sim_bio_proc_reg.sample(frac=1, random_state=random_state)

# set data types
sim_bio_proc_reg = sim_bio_proc_reg.astype('float64')

# log10 of targets -ws, -a, and -f
sim_bio_proc_reg['-ws'] = np.log10(sim_bio_proc_reg['-ws'])
sim_bio_proc_reg['-a'] = np.log10(sim_bio_proc_reg['-a'])
sim_bio_proc_reg['-f'] = np.log10(sim_bio_proc_reg['-f'])

# name dataset
sim_bio_proc_reg.name = 'Train_11000 Regression'

# display dataset
print(sim_bio_proc_reg.name)
print(sim_bio_proc_reg.shape, '\n')
display(sim_bio_proc_reg.head())
display(sim_bio_proc_reg.iloc[sim_bio_proc_reg.shape[0]-5:sim_bio_proc_reg.shape[0], :])

# dataset for each target
sim_bio_proc_reg_ws = pd.concat((sim_bio_proc_reg.iloc[:,0:-3], sim_bio_proc_reg['-ws']), axis=1)
sim_bio_proc_reg_a = pd.concat((sim_bio_proc_reg.iloc[:,0:-3], sim_bio_proc_reg['-a']), axis=1)
sim_bio_proc_reg_f = pd.concat((sim_bio_proc_reg.iloc[:,0:-3], sim_bio_proc_reg['-f']), axis=1)
sim_bio_proc_reg_ws.name = 'Train_11000 Regression -ws'
sim_bio_proc_reg_a.name = 'Train_11000 Regression -a'
sim_bio_proc_reg_f.name = 'Train_11000 Regression -f'

print('\n', sim_bio_proc_reg_ws.name)
print(sim_bio_proc_reg_ws.shape)
display(sim_bio_proc_reg_ws.head())

print('\n', sim_bio_proc_reg_a.name)
print(sim_bio_proc_reg_a.shape)
display(sim_bio_proc_reg_a.head())

print('\n', sim_bio_proc_reg_f.name)
print(sim_bio_proc_reg_f.shape)
display(sim_bio_proc_reg_f.head())
