# -*- coding: utf-8 -*-


import os
import json
import requests


def call_api_upload_model(path,file_name,url = 'http://localhost:5000/upload_model'):
    
    """
    
    """
    
    file_pickle = os.path.join(path,file_name + '.pickle')
    file_py = os.path.join(path,file_name + '.py')
    
    params = {'filename' : file_name}
    upload_files = {'file_pickle':open(file_pickle,'rb'),
                    'file_py':open(file_py,'r',encoding = 'utf-8')}
    
    res = requests.post(url,data = params,files = upload_files)
    
    
    return res.text



def call_api_invoke_model(model_name,args:dict,url = 'http://localhost:5000/call_model'):
    """
    
    """
    
    params = {'model_name':model_name}
    headers = {'content-type':'application/json'}
    res = requests.post(url,headers = headers,data = json.dumps({'params':params,'args':args}))
    
    return res.text


