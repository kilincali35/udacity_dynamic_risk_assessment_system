"""
This will run predictions, get score, and create summary stats and diagnostic report for this test run.

In this version, it is using a fixed test_set located in test directory.

Maybe it is a good idea to use same test set for various models, and drift check for a long time window. 

author: ali.kilinc
date: 07/02/23
"""

import json
import os
import time
import pandas as pd
import requests

URL = "http://127.0.0.1:8000"


def main():
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = config['test_data_path']
    model_path = config['output_model_path']

    test_df = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')
    test_df_json = test_df.to_json(orient = 'table', index = False)
    
    headers = {'Content-Type': 'application/json'}
    
    print('Reaching /prediction via apicalls')
    predictions = requests.post(URL + '/prediction', test_df_json, headers=headers)
    print('Predictions from apicalls')
    print(predictions)
    print(predictions.json())

    print('Reaching /scoring from apicalls')
    score = requests.get(URL + '/scoring')
    print(score.json())

    print('Reaching /summarystats from apicalls')
    summary = requests.get(URL + '/summarystats')
    print(summary.json())

    print('Reaching /diagnostics from apicalls')
    diagnosis = requests.get(URL + '/diagnostics')
    print(diagnosis.json())

    response = {
        'predictions': predictions.json(),
        'score': score.json(),
        'summary': summary.json(),
        'diagnosis': diagnosis.json()
    }

    output_file = os.path.join(os.getcwd()+model_path, f'apireturns_{time.strftime("%y%m%d%H%M%S")}.txt')
    
    with open(output_file, 'w') as f:
        print(f'Writing API responses to {output_file}')
        f.write(json.dumps(response, indent=4))


if __name__ == '__main__':
    main()


