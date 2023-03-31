"""
Script to define the moitoring and reporting API serving the trained model.

Author: Paulo Souza
Date: Mar 2023
"""
from flask import Flask, request
from diagnostics import *
from scoring import score_model
import json
import os


app = Flask(__name__)
#app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)
dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

@app.route("/prediction", methods=['GET', 'POST','OPTIONS'])
def predict():
    '''
    Returns predictions from the deployed model over the given test data.

    Returns
        preds: List
            A list containing the model predictions for the given test data.
    '''
    data_path = request.args.get('data_path')
    print(os.getcwd())
    preds = model_predictions(data_path=data_path)
    return preds

@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    '''
    Returns the scoring performance obtained by the depoloyed model.

    Returns
        f1: float
            F1-score obtained by the trained model over the test data.
    '''
    return str(score_model())

@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    '''
    Checks means, medians, and std for each numeric column.

    Returns
        statistic_list: List[List]
            A list containing the summary statistics of each numeric column
            present in the given dataset. The metrics are displayed in the
            following order: [[mean, median, std],...]
    '''
    return dataframe_summary()

@app.route("/diagnostics", methods=['GET','OPTIONS'])
async def diagnostics():
    '''
    Checks timing and percent NA values.

    Returns
        data_integrity: List
            The percentages of missing observations in each numeric column of
            the given dataset.
        timings: List
            A list with the timings of each process step, in seconds.
    '''
    data_integrity = check_data_integrity()
    timings = execution_time()
    dependencies = await async_outdated_packages_list()

    return [
        data_integrity,
        '-----------------------',
        timings,
        '-----------------------',
        dependencies
    ]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
