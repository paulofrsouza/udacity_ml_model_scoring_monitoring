'''
Script for generating a confusion matrix plot over the trained model's results.

Author: Paulo Souza
Date: Mar 2023
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os
from diagnostics import model_predictions

with open('config.json','r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])

def score_model(
    data_path: str = test_data_path,
    model_path: str = prod_deployment_path
) -> None:
    '''
    Calculates a confusion matrix using the test data and the deployed model.
    Writes the confusion matrix to the workspace.

    Args:
        data_path: str
            A path indicating where to look for test datasets.
        model_path: str
            A path indicating where to look for trained models.
    '''
    preds = model_predictions(data_path, model_path)

    test_data = [el for el in os.listdir(data_path) if '.csv' in el]
    final = pd.DataFrame()
    for data in test_data:
        temp = pd.read_csv(os.path.join(test_data_path, data))
        final = pd.concat([final, temp], axis=0)
    final.drop_duplicates(inplace=True)
    final.drop('corporation', axis=1, inplace=True)

    x_test = final.drop('exited', axis=1)
    y_test = final['exited']

    conf = confusion_matrix(y_test, preds)
    with open(
        os.path.join(output_model_path, 'confusionmatrix.txt'),
        'w'
    ) as f:
        f.write(np.array_str(conf))

if __name__ == '__main__':
    score_model()
