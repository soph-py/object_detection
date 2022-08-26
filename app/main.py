import requests, pickle
import pandas as pd
from src.client import Client
from src.postprocess import Postprocess

""" 
:NOTE: 
    a random sample of 10 addresses was used to test the functionality of the preprocess.py and client.py modules
    the image bytes for each of the 10 addresses were saved in a list in the pickle file called 'img_bytes_list.pkl'
    the commented out code below can be skipped over for now up until 'multiple_files' is defined
"""

# ## a copy of the google sheet containing addresses with relevant columns only
# df = pd.read_csv('static/address_data.csv')
# df.head()
# df.columns # 'id', 'applied_ml', 'street', 'city', 'state', 'zipcode', 'has_pool', 'has_solar_panels'

# # applied_ml == True if observation was used in train,val,test split during transfer learning of yolov5
# # we want to test on unseen data, so filter rows where applied_ml == False
# df_unseen = df.loc[df['applied_ml'] == False].copy()
# df_sample = df_unseen.sample(n=10, replace=False) # randomly sample 10 observations/addresses

# test = dict()
# for ind in df_sample.index:
#     key = str(df_sample['id'][ind]) # use unique identifier "id" as dict key from each of the 10 samples 
#     value = dict(
#         street=df_sample['street'][ind],
#         city=df_sample['city'][ind],
#         state=df_sample['state'][ind]
#         )
#     if isinstance(str('1'), type(df_sample['zipcode'][ind])): # zipcode exists for this row, add it to the test dict
#         value['zipcode'] = df_sample['zipcode'][ind]
#     test[key] = value

# # create an instance of the Client() preprocessing class, which will return image as bytes for each value in test dict
# img_bytes = [] ## list of image bytes that we will use to send a post request
# for value in test.values():
#     c = Client(**value)
#     response = c._request()
#     img_bytes.append(response.content)

# # send unique identifier for each address to api as "data" param
# data = dict(index=list(test.keys())) 


with open('static/img_bytes_list.pkl', 'rb') as file:
    img_bytes = pickle.load(file)

with open('static/data_index.pkl', 'rb') as file:
    data = pickle.load(file)

# send a tuple of ('files', image bytes) for each address in a list - this is how flask allows multiple file uploads
multiple_files = [('files', file) for file in img_bytes]

""" 
POST request headers: {'Server': 'Werkzeug/2.2.1 Python/3.8.13', 
                        'Content-Type': 'application/json'}
"""

r = requests.post('http://127.0.0.1:5000/', files = multiple_files, data = data)
response = r.json()

postprocess = Postprocess()
postprocess.detection_map = (response['index'], response['detections'])
postprocess.updated_detections() # predictions mapped back to address's unique ID