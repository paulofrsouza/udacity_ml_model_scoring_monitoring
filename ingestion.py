'''
Ingests, process and persists input data present in the folders described in
the config.json file.

Author: Paulo Souza
Date: Mar 2023
'''

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

with open('config.json','r') as f:
    config = json.load(f)
current_folders = os.listdir()

input_folder_path = config['input_folder_path']
if input_folder_path not in current_folders:
    raise Exception(
        f"""\
        {input_folder_path} folder not present in the project directory.
        Please indicate a valid folder.
        """
    )

output_folder_path = config['output_folder_path']
if output_folder_path not in current_folders:
    os.mkdir(output_folder_path)

def merge_multiple_dataframes() -> None:
    '''
    Checks for datasets, compile them together, and write to an output file
    '''
    input_data = [el for el in os.listdir(input_folder_path) if '.csv' in el]
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(', '.join(input_data))

    final = pd.DataFrame()
    for data in input_data:
        temp = pd.read_csv(os.path.join(input_folder_path, data))
        final = pd.concat([final, temp], axis=0)

    final.drop_duplicates(inplace=True)
    final.to_csv(
        os.path.join(output_folder_path, 'finaldata.csv'),
        index=False
    )

if __name__ == '__main__':
    merge_multiple_dataframes()
