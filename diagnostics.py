"""
Script to run diagnostics over the generated artifacts and project steps in the
project.

Author: Paulo Souza
Date: Mar 2023
"""

import pandas as pd
import timeit
import os
import json
from typing import List
from textwrap import dedent
import subprocess
import pickle

with open('config.json','r') as f:
    config = json.load(f)
current_folders = os.listdir()

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

def model_predictions() -> List:
    '''
    Reads the deployed model and a test dataset, calculates predictions

    Returns:
        preds: List
            A list containing the model predictions for the given test data.
    '''
    with open(
        os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'
    ) as f:
        lr = pickle.load(f)

    test_data = [el for el in os.listdir(test_data_path) if '.csv' in el]
    final = pd.DataFrame()
    for data in test_data:
        temp = pd.read_csv(os.path.join(test_data_path, data))
        final = pd.concat([final, temp], axis=0)
    final.drop_duplicates(inplace=True)
    final.drop('corporation', axis=1, inplace=True)

    x_test = final.drop('exited', axis=1)
    y_test = final['exited']

    preds = lr.predict(x_test)
    return preds

def dataframe_summary() -> List[List]:
    '''
    Calculates summary statistics from the given dataset. Writes the summary
    statics report in the dataset's folder.

    Returns
        statistic_list: List[List]
            A list containing the summaty statistics of each numeric column
            present in the given dataset. The metrics are displayed in the
            following order: [[mean, median, std],...]
    '''

    summ_data = [el for el in os.listdir(dataset_csv_path) if '.csv' in el]
    final = pd.DataFrame()
    for data in summ_data:
        temp = pd.read_csv(os.path.join(dataset_csv_path, data))
        final = pd.concat([final, temp], axis=0)
    final.drop_duplicates(inplace=True)
    final.drop('corporation', axis=1, inplace=True)

    report = ''
    statistic_list = []
    for col in final.columns.tolist():
        mean = final[col].mean()
        median = final[col].median()
        std = final[col].std()
        metrics = [mean, median, std]
        statistic_list.append(metrics)

        report += dedent(
            f"""\
            Variable: {col}
            ---------------
            Mean: {mean}
            Median: {median}
            Std: {std}

            """
        )

    with open(os.path.join(dataset_csv_path, 'summary_metrics.txt'), 'w') as f:
        f.write(report)

    return statistic_list

def check_data_integrity() -> None:
    '''
    Checks for dataset data integrety by measuring the percentage of missing
    datapoints in each numeric column.

    Returns
        data_integrity: List
            The percentages of missing observations in each numeric column of
            the given dataset.
    '''

    summ_data = [el for el in os.listdir(dataset_csv_path) if '.csv' in el]
    final = pd.DataFrame()
    for data in summ_data:
        temp = pd.read_csv(os.path.join(dataset_csv_path, data))
        final = pd.concat([final, temp], axis=0)
    final.drop_duplicates(inplace=True)
    final.drop('corporation', axis=1, inplace=True)

    report = ''
    data_integrity = []
    for col in final.columns.tolist():
        missing_perc = round(final[col].isna().sum() / final.shape[0], 2)
        data_integrity.append(missing_perc)

        report += dedent(
            f"""\
            Variable: {col}
            ---------------
            Missing data percentage: {missing_perc} %

            """
        )

    with open(os.path.join(dataset_csv_path, 'data_integrity.txt'), 'w') as f:
        f.write(report)

    return data_integrity

def execution_time() -> List:
    '''
    Calculates the timings of ingestion.py and training.py.
    
    Returns
        timings: List
            A list with the timings of each process step, in seconds.
    '''

    ingestion_start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_process_time = timeit.default_timer() - ingestion_start_time

    training_start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_process_time = timeit.default_timer() - training_start_time

    return [ingestion_process_time, training_process_time]

def outdated_packages_list() -> None:
    '''
    Fetches all modules being used in the project and checks for outdated
    dependencies.
    '''
    with open('./requirements.txt', 'r') as f:
        reqs = f.readlines()

    reqs = [el[:el.find('=')] for el in reqs]

    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    outdated = str(outdated, 'utf-8')
    outdated = [el.split(' ') for el in outdated.split('\n')]
    outdated = [[s for s in el if s != ''] for el in outdated][:-1]
    del outdated[1]
    outdated = [el[:-1] for el in outdated]
    outdated = pd.DataFrame(data=outdated[1:], columns=outdated[0])
    outdated.set_index('Package', inplace=True)

    uptodate = subprocess.check_output(['pip', 'list', '--uptodate'])
    uptodate = str(uptodate, 'utf-8')
    uptodate = [el.split(' ') for el in uptodate.split('\n')]
    uptodate = [[s for s in el if s != ''] for el in uptodate][:-1]
    del uptodate[1]
    uptodate = pd.DataFrame(data=uptodate[1:], columns=uptodate[0])
    uptodate.set_index('Package', inplace=True)
    uptodate['Latest'] = uptodate['Version']

    final = pd.concat([outdated, uptodate], axis=0)
    final.loc[reqs, :].to_csv('./dependency_check.csv')

if __name__ == '__main__':
    preds = model_predictions()
    statistic_list = dataframe_summary()
    data_integrity = check_data_integrity()
    exec_time = execution_time()
    outdated_packages_list()
