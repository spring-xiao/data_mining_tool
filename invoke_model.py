# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:15:05 2020

@author: xiaoxiang
"""

import pandas as pd
import numpy as np
import pickle
import requests
import json
import os

from src.file_utils import read_pickle_file,load_file_py


os.chdir("D:\\Spyder\data_mining_tool")


code = load_file_py('model/A001V001120200601192955.py')
exec(code)


model_info = read_pickle_file('model/A001V001120200601192955.pickle')


x_var = model_info.get('x_var')
var_pipe = model_info.get('var_pipe')
model = model_info.get('model')


input_data = pd.read_csv('data/accepts.csv')
input_data_pro = var_pipe.transform(input_data[x_var])
model.predict_proba(input_data_pro)


#******************************************************************************


from api.call_api import call_api_invoke_model,call_api_upload_model
from src.utils import transf_data_type

args = transf_data_type(input_data.iloc[2])
model_name = 'A001V001120200601192955'

call_api_invoke_model(model_name,args)











