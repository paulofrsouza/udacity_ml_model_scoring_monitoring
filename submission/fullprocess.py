'''
Automates the whole model pipeline.

Author: Paulo Souza
Date: Mar 2023
'''

import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
from typing import Union, List
import pickle
from textwrap import dedent

with open('config.json','r') as f:
    config = json.load(f)
curr_folders = os.listdir()

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
if input_folder_path not in curr_folders:
    os.mkdir(input_folder_path)
output_model_path = os.path.join(config['output_model_path'])
if output_model_path not in curr_folders:
    os.mkdir(output_model_path)

def check_new_data() -> Union[bool, List[str]]:
    '''
    Reads ingestedfiles.txt and determines whether the source data folder has
    files that aren't listed in ingestedfiles.txt.
    
    Returns
        new_data_flag:
            Bool indicating wheter there is new data or not.
        new_data_list:
            List containing the new data files found in the input directory.
    '''
    new_data_flag = False
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
        ingested_files = f.read()
    ingested_files = ingested_files.split(', ')
    curr_files = [el for el in os.listdir(input_folder_path) if '.csv' in el]
    new_data_list = []
    for file in curr_files:
        if file not in ingested_files:
            new_data_flag = True
            new_data_list.append(file)

    return new_data_flag, new_data_list

def check_model_drift(
    data_path: str = input_folder_path,
    model_path: str = prod_deployment_path
) -> bool:
    '''
    Checks whether there is model drift between datasets or not.
    
    Args:
        data_path: str
            A path indicating where to look for test datasets.
        model_path: str
            A path indicating where to look for trained models.
    Returns
        model_drift_flag:
            Bool indicating the presence, or not, of model drift.
    '''
    model_drift_flag = False
    
    with open(os.path.join(model_path, 'latestscore.txt'), 'r') as f:
        f1_old = f.read()
    f1_old = float(f1_old)

    f1_new = scoring.score_model(
        data_path=data_path,
        model_path=model_path
    )

    if f1_old < f1_new:
        model_drift_flag = True

    return model_drift_flag

def pipeline() -> None:
    '''
    Fully automated model pipeline
    '''
    new_data_flag, new_data_list = check_new_data()
    if not new_data_flag:
        return 'No new data was found. Ending pipeline.'
    else:
        print('New data was found. Proceeding to check for model drift.')

    model_drift_flag = check_model_drift()
    if not model_drift_flag:
        return 'No model drift detected. Ending pipeline.'
    else:
        print(dedent(
            """\
            Model drift was found. Proceeding to re-train and re-deploy the
            model.
            """
        ))

    os.system('python3 ingestion.py')
    os.system('python3 training.py')
    os.system('python3 deployment.py')
    os.system('python3 diagnostics.py')
    os.system('python3 reporting.py')
    os.system('python3 apicalls.py')
    print('Pipeline process completed!')


if __name__ == '__main__':
    pipeline()




