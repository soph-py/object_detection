import requests, pickle
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from src.client import Client
from src.postprocess import Postprocess

## user defined types for type hinting support
GenericNumber = Union[int, float]
Key, Value = Union[str, int], Union[str, GenericNumber]
Prediction = Dict[Key, Value] # singular prediction
Detections = Dict[str, List[Prediction]] # list of predicted objects detected per image
IndexMap = List[Union[str, int]]
BytesFiles = List[Tuple[str, bytes]]

def load_df_from_csv():
    pass

def post_files(df_dict) -> Tuple[IndexMap, BytesFiles]: # Tuple[List[str],BytesFiles]
    """ 
    incoming_request_files to post in request
    """
    # df_dict needs to be in this format with index of observations as keys
    # test1 = dict(street='2301 BEAMREACH CT', city='lincoln', state='CA', zipcode='95648')
    # test2 = dict(street='3731 CEDARGATE WAY', city='SACRAMENTO', state='CA')
    img_bytes = [] # list of bytes objects
    for n in df_dict.values():
        client = Client(**n)
        response = client.response()
        img_bytes.append(response.content) # appends bytes content of an image to img_bytes list
    return (
        list(df_dict.keys()), 
        [('files', file) for file in img_bytes]
        )

def request() -> Dict[str, Union[IndexMap, Detections]]:
    files = post_files()
    data = dict(index=files[0]) # df unique id number
    multiple_files = files[1]
    response = requests.post('http://127.0.0.1:5000/', files=multiple_files, data=data)
    return response.json()

def postprocess_predictions():
    postprocess = Postprocess()
    response = request()
    postprocess.detection_map = (response['index'], response['detections'])
    obj_index_map = postprocess.updated_detections() # predictions mapped back to address's unique ID
    return obj_index_map

def main():
    # take cmd line args, i.e. path to csv file
    # then call functions in the order below:
    load_df_from_csv()
    pass

if __name__ == '__main__':
    ...