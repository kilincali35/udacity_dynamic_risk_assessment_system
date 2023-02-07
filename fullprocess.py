"""
This process is for checking newcoming data feed. In case of a new data, it traings new model with new dataset. 

If new model is an improved version of production model, then it automatically gets deployed into production. 

All the diagnostics and reporting is created afterwards. 

*** It is using a basic, static version of retrieving datasets. We will need to transform it directly from DB, rather than static csv files,
for real life purposes. This one is a basic version, used in Udacity project

author: ali.kilinc
date: 07/02/23

"""

from training import main as train
from scoring import main as score
from ingestion import main as ingest
from apicalls import main as apicalls
from deployment import main as deploy
from diagnostics import main as diagnose
from reporting import main as report
import glob
import json
import os
from typing import Tuple

def check_new_files(dataset_path, deployment_path):
    """
    This function looks at the deployment_path, to get datasets used in deployed model. 
    
    Then checks sourcedata folder, to compare, if there is a new data feed.
    
    *** In this version, it gets all file names from sourcedata folder. There must be only csv files. To be errorproof, it can be converted into
        getting only .csv files from that folder)
    """
    
    print('Checking for new files...')
    with open(os.getcwd()+deployment_path+'ingestedfiles.txt', 'r') as f:
        ingested_files = f.read().splitlines()

    files = set(os.listdir(os.getcwd()+dataset_path))
    new_files = [f for f in files if f not in ingested_files]
    return new_files


def check_model_drift(deployment_path):
    """
    This function will look at the latest model score from deployed_model path. Then runs the training script to get new score, with new data. 
    
    It will compare new score to old one, and pushes score as an output. 
    """

    print('Checking for model drift...')
    with open(os.getcwd()+deployment_path+'latestscore.txt', 'r') as f:
        old_f1_score = float(f.readline().strip())

    train()
    new_f1_score = score()
    print(new_f1_score)

    return new_f1_score > old_f1_score, new_f1_score, old_f1_score


def main():
    """
    In the main process, it checks for new files, if not, then stops the process.
    
    If there are any new files, it will ingest new files and run training to check model drift, if it finds a better model, process will go on.
    
    It will deploy the model, run diagnostics. 
    
    At the end, it will generate diagnostic reports based on a test data automatically, based on an API we created at app.py, via apicalls.py script. 
    """
    print("Running fullprocess...")
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    deployment_path = config['prod_deployment_path']
    model_path = config['output_model_path']

    
    if  len(check_new_files(input_folder_path, deployment_path)) == 0:
        print(f'No new dataset in {input_folder_path}. Ending process...')
        exit()
    
    ingest()

    drift, new_score, old_score = check_model_drift(deployment_path)
    
    if not drift:
        print('Production model performs better. '
              f'New F1-Score: {new_score}. Old F1-Score: {old_score}')
        exit()

    deploy()
    report()
    apicalls()


if __name__ == '__main__':
    main()