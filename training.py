'''
Trains a Logistic Regression with the processed data available.

Author: Paulo Souza
Date: Mar 2023
'''

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

with open('config.json','r') as f:
    config = json.load(f)
current_folders = os.listdir()

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
if model_path not in current_folders:
    os.mkdir(model_path)

def train_model() -> None:
    '''
    Trains the model and persists a serialized version of it.
    '''

    lr = LogisticRegression(
        C=0.1,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        #l1_ratio=0.75,
        max_iter=200,
        multi_class='auto',
        n_jobs=1,
        penalty='l2',
        random_state=42,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False
    )

    train_data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    train_data.drop('corporation', axis=1, inplace=True)

    x_train = train_data.drop('exited', axis=1)
    y_train = train_data['exited']
    
    lr.fit(x_train, y_train)
    
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(lr, f)

if __name__ == '__main__':
    train_model()