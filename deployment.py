"""
This will carry the new production model to deployed model path, together with the logs.

author: ali.kilinc
date: 07/02/23
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import shutil
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle(model_path, prod_deployment_path, dataset_csv_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copyfile(os.getcwd()+model_path+'trainedmodel.pkl', os.getcwd()+prod_deployment_path+'trainedmodel.pkl')
    shutil.copyfile(os.getcwd()+dataset_csv_path+'ingestedfiles.txt', os.getcwd()+prod_deployment_path+'ingestedfiles.txt')
    shutil.copyfile(os.getcwd()+model_path+'latestscore.txt', os.getcwd()+prod_deployment_path+'latestscore.txt')
        
def main():
    store_model_into_pickle(model_path, prod_deployment_path, dataset_csv_path)
    
if __name__ == '__main__':
    main()
