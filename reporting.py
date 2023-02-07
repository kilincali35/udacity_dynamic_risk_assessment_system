"""
This module will create a model report with required metrics.

In this version, it only saves a confusion matrix.

author: ali.kilinc
date: 07/02/23
"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

df = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')

##############Function for reporting
def score_model(df, model_path):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    y_preds, df_y = model_predictions(df, model_path)
    
    # plot the confusion matrix
    confusion = metrics.confusion_matrix(df_y, y_preds)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(confusion, annot=True, cmap='Blues')

    ax.set_title('Confusion matrix')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(os.getcwd()+model_path+'/confusionmatrix.png')

def main():
    score_model(df, model_path)

if __name__ == '__main__':
    main()
