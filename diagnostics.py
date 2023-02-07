"""
This module has severaş important tools for model and data diagnostics. 

author: ali.kilinc
date: 07/02/23
"""
import pandas as pd
import numpy as np
import timeit
import pickle
import subprocess
import os
import glob
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

df = pd.read_csv(os.getcwd()+dataset_csv_path+'finaldata.csv', index_col=[0])

##################Function to get model predictions
def model_predictions(df, model_path):

    df_x = df.drop(['corporation', 'exited'], axis=1)
    df_y = df['exited']
    
    with open(os.getcwd()+model_path+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(df_x)
    return y_pred, df_y

##################Function to get summary statistics
def dataframe_summary(dataset_csv_path, index_col):
    
    #This one goes to a directory, and finds the latest csv file in there
    latest_csv = max(glob.glob(os.getcwd()+dataset_csv_path+'*.csv'), key=os.path.getctime)

    data_df = pd.read_csv(latest_csv, index_col=index_col)
    data_df = data_df.drop(['corporation', 'exited'], axis=1)

    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()

        statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics_dict

def missing_data(dataset_csv_path, index_col):
    
    #This one goes to a directory, and finds the latest csv file in there
    latest_csv = max(glob.glob(os.getcwd()+dataset_csv_path+'*.csv'), key=os.path.getctime)
    
    data_df = pd.read_csv(latest_csv, index_col=index_col)
    missing = data_df.isna().sum()
    n_data = data_df.shape[0]
    missing = missing / n_data
    print(missing)
    return missing.to_dict()

##################Function to get timings
def execution_time():
    
    # timing ingestion
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing = timeit.default_timer() - starttime

    # timing training
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - starttime

    return [ingestion_timing, training_timing]

##################Function to check dependencies
def outdated_packages_list():

    # current version of dependencies
    with open(os.getcwd() + '/requirements.txt', 'r') as req_file:
        requirements = req_file.read().split('\n')
        
    requirements = [r.split('==') for r in requirements if r]
    df = pd.DataFrame(requirements, columns=['module', 'current'])

    # Get outdated dependencies using PIP
    outdated_dep = subprocess.check_output(['pip', 'list', '--outdated']).decode('utf8')
    outdated_dep = outdated_dep.split('\n')[2:]  # the first 2 items are not packages
    outdated_dep = [x.split(' ') for x in outdated_dep if x]
    outdated_dep = [[y for y in x if y] for x in outdated_dep]  # list of [package, current version, latest version]
    outdated_dic = {x[0]: x[2] for x in outdated_dep}  # {package: latest version}
    df['latest'] = df['module'].map(outdated_dic)

    # if we're already using the latest version of a module, we fill latest with this version:
    df['latest'].fillna(df['current'], inplace=True)
    return df.to_dict('records')

def main():
    y_pred = model_predictions(df, model_path)
    
    stats = dataframe_summary(dataset_csv_path, index_col = [0])
    print("Column statistics:")
    print(stats)
    print(".................")
    
    print("Missing data:")
    missing_data(dataset_csv_path, index_col = [0])
    print("................")
    
    print("Time Check")
    time_check = execution_time()
    print(time_check)
    print(".............")
    
    print("Outdated list:")
    outdated = outdated_packages_list()
    print(outdated)
    pass

if __name__ == '__main__':
    main()





    
