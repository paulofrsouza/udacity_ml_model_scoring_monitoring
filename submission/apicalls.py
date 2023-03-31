'''
Calls each created API endpoint in the project.

Author: Paulo Souza
Data: Mar 2023
'''
import requests
import json
import os

with open('config.json','r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])

URL = "http://127.0.0.1:8000"

def call_api(output_path: str = output_model_path) -> None:
    response1 = requests.post(os.path.join(URL, 'prediction?data_path=testdata')).text
    response2 = requests.get(os.path.join(URL, 'scoring')).text
    response3 = requests.get(os.path.join(URL, 'summarystats')).text
    response4 = requests.get(os.path.join(URL, 'diagnostics')).text

    responses = '\n-----------------------\n'.join(
        [response1, response2, response3, response4]
    )

    with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
        f.write(responses)

if __name__ == '__main__':
    call_api()