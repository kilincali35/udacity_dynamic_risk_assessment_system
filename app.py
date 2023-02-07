"""
This file is a Flask API file, to automate the process and share the results via an API.

It will run predictions on a data pushed by apicalls. 

Then it will prepare score, diagnostics and summary stats for the new model recently deployed. 

author: ali.kilinc
date: 07/02/23
"""

from flask import Flask, request, jsonify
import pandas as pd
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576nkl1vjh224jfn21g'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    print('Running /prediction from App')
    json_data = request.get_json()
    df = pd.DataFrame(json_data['data'])

    predictions, df_y = model_predictions(df, prod_deployment_path)

    return json.dumps(predictions.tolist()), 200

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    
    print('Running /scoring from App')
    score = score_model(prod_deployment_path, test_data_path)

    return str(score), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    
    print('Running /summarystats from App')
    col_stats = dataframe_summary(test_data_path, index_col = False)
    return jsonify(col_stats), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    
    print('Running /diagnostics from App')
    missing = missing_data(test_data_path, index_col = False)
    time_check = execution_time()
    outdated = outdated_packages_list()
    diag_dict = {'missing': missing, 'time_check': time_check, 'outdated': outdated}
    return jsonify(diag_dict)

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=False, threaded=True)
