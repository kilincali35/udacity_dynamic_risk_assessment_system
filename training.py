"""
This module trains the model and saves it as a pickle file.

For this project with very small data, one fixed model is used for training. 

There will be a more complex structure here, like gridsearchcv for different kind of algorithms. 

author:ali.kilinc
date 07/02/23
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

train_data = pd.read_csv(os.getcwd()+dataset_csv_path+'finaldata.csv', index_col=[0])
train_df = pd.DataFrame(train_data)

df_y = train_df['exited']
df_x = train_df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

#################Function for training the model
def train_model(x,y):
    
    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    #fit the logistic regression to your data
    print('A new model is fitting on a dataset with length:' + str(len(x)))
    model = logit.fit(x,y.values.reshape(-1,1).ravel())
    print('Model is fitted succesfully with new data')
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(os.getcwd()+model_path+'trainedmodel.pkl', 'wb'))
    print('Model is saved to' + str(os.getcwd()+model_path+'trainedmodel.pkl'))

def main():
    train_model(df_x, df_y)
    
if __name__ == '__main__':
    main()