"""
Script for scoring the Logisc Regression model trained in the project.

Author: Paulo Souza
Date: Mar 2023
"""

import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import json

with open('config.json','r') as f:
    config = json.load(f)
current_folders = os.listdir()

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path'])
if test_data_path not in current_folders:
    raise Exception(
        f"""\
        {input_folder_path} folder not present in the project directory.
        Please indicate a valid folder.
        """
    )

def score_model() -> float:
    """
    Takes a trained model, loads test data, and calculates an F1 score for the
    given model relative to the test data. Lastly, it writes the result to the
    latestscore.txt file.
    
    Returns:
        f1: float
            F1-score obtained by the trained model over the test data.    
    """

    test_data = [el for el in os.listdir(test_data_path) if '.csv' in el]
    final = pd.DataFrame()
    for data in test_data:
        temp = pd.read_csv(os.path.join(test_data_path, data))
        final = pd.concat([final, temp], axis=0)
    final.drop_duplicates(inplace=True)
    final.drop('corporation', axis=1, inplace=True)

    x_test = final.drop('exited', axis=1)
    y_test = final['exited']

    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        lr = pickle.load(f)

    preds = lr.predict(x_test)
    print(preds)
    f1 = f1_score(y_test, preds)
    scoring = f"Trained model's F1-score: {f1}"

    with open(os.path.join(model_path, "latestscore.txt"), 'w') as f:
        f.write(scoring)
        
    return f1


if __name__ == '__main__':
    f1 = score_model()
