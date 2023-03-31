"""
Script for 'deploying' the trained model to a production environment.

Author: Paulo Souza
Date: Mar 2023
"""

import os
import json
from shutil import copy2

with open('config.json','r') as f:
    config = json.load(f)
current_folders = os.listdir()

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
if prod_deployment_path not in current_folders:
    os.mkdir(prod_deployment_path)

def store_model_into_pickle() -> None:
    """
    Copies the latest trained model pickle file, the latestscore.txt value and
    the ingestfiles.txt file into the deployment directory.
    """

    copy2(
        os.path.join(model_path, 'trainedmodel.pkl'),
        os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    )
    copy2(
        os.path.join(model_path, "latestscore.txt"),
        os.path.join(prod_deployment_path, "latestscore.txt")
    )
    copy2(
        os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
        os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    )


if __name__ == '__main__':
    store_model_into_pickle()
