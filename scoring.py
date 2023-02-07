"""
This module is used to create metric scores and record them as a log for a production model

author: ali.kilinc
date: 07/02/23
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
from datetime import datetime

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(model_path, test_data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_df = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')
    test_df_y = test_df['exited']
    test_df_x = test_df[['lastmonth_activity','lastyear_activity','number_of_employees']]
    
    with open(os.getcwd()+model_path+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    predicted = model.predict(test_df_x)
    f1score = metrics.f1_score(predicted, test_df_y)
    
    return f1score

dateTimeObj = datetime.now()
thetimenow = str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

#f1score = score_model(model_path, test_data_path)

def record_f1score(f1score,model_path):
    
    #Using a for open module means append, not overwrite
    MyFile = open(os.getcwd()+model_path+'score_logs.txt','a')
    MyFile.write('.........' + thetimenow + '...............')
    MyFile.write('\n')
    MyFile.write('F1_Score:')
    MyFile.write(str(f1score))
    MyFile.write('\n')
    MyFile.write('\n')
    MyFile.write('........................')
    MyFile.close()
    
    MyFile2 = open(os.getcwd()+model_path+'latestscore.txt', 'w')
    MyFile2.write(str(f1score))
    MyFile2.close()
    
def main():
    
    f1score = score_model(model_path, test_data_path)
    record_f1score(f1score, model_path)
    
    return f1score
    
if __name__ == '__main__':
    main()