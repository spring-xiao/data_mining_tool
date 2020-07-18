# -*- coding: utf-8 -*-

import time
import pickle


def generate_model_file_name(model_type:str,model_version:str,model_alg:str,model_time = None,file_type = 'pickle'):
    
    if model_time is None:
        model_time = time.strftime("%Y%m%d%H%M%S",time.localtime())

    model_file_name = str(model_type) + \
                      str(model_version) + \
                      str(model_alg) + \
                      str(model_time)
    
    if file_type is not None:
        model_file_name = model_file_name + '.' + file_type
    
    return model_file_name


def save_pickle_file(file_path,obj,is_print = True):
    
    with open(file_path,'wb') as file:
        pickle.dump(obj,file)
    
    if is_print:
        print('保存成功')


def read_pickle_file(file_path):
    
    with open(file_path,'rb') as file:
       obj = pickle.load(file)
     
    return obj


def load_file_py(filename): 
    
    with open(filename,'r',encoding = 'utf-8') as file:
        code = file.read()
        
    return code


