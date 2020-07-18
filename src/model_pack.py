# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:25:28 2020

@author: xiaoxiang
"""

from src.files_utils import generate_model_file_name,save_pickle_file
import inspect
import time
import os
from sklearn.pipeline import Pipeline


self_cls_module = ['src.var_processor',
                   'src.var_filter']

#******************************************************************************
def get_all_cls(cls,cls_module:list = self_cls_module):
    
    all_cls = []
    all_cls.append(cls)
    def get_base_cls(cls,all_cls:list):
    
        for supercls in list(cls.__bases__):
            if supercls.__module__ in cls_module:
                all_cls.append(supercls)
            
            all_cls = get_base_cls(supercls,all_cls)
    
        return all_cls
    
    res = get_base_cls(cls,all_cls)
    res.reverse()
    
    return res


def get_obj_code(obj):
    
    lines = inspect.getsourcelines(obj)
    lines = ''.join(lines[0])
    
    return lines
    

def get_obj_lst_code(obj_lst:list):
    
    lines_lst = [get_obj_code(obj) for obj in obj_lst]
    return lines_lst


#******************************************************************************

from src.basic_utils import basic_utils_fun
from src.var_bin_utils import var_bin_utils_fun
#from src.utils import utils_fun

public_import_code = ['from scipy.stats import chi2,skew\n',
                      'from sklearn.base import BaseEstimator,TransformerMixin\n',
                      'from sklearn.tree import  DecisionTreeClassifier as DTree\n'
                      'from sklearn.preprocessing import (\n',
                      'LabelEncoder,\n',
                      'OneHotEncoder,\n',
                      'StandardScaler,\n',
                      'MinMaxScaler)\n',
                      'import pandas as pd\n',
                      'import numpy as np\n',
                      'import re\n',
                      'import copy\n']


public_obj_lst = list(basic_utils_fun.values()) + \
                 list(var_bin_utils_fun.values())


class ModelPack(object):
    
    
    def __init__(self,model_info:dict = None,model_type:str = 'T001',model_version:str = 'V001',model_alg:str = 'A1',model_time = None):
        
        self.model_info = model_info
        self.model_type = model_type
        self.model_version = model_version
        self.model_alg = model_alg
        self.model_time = time.strftime("%Y%m%d%H%M%S",time.localtime()) if model_time is None else model_time
        self.model_file_pickle = generate_model_file_name(self.model_type,self.model_version,self.model_alg,self.model_time,'pickle')
        self.model_file_py = generate_model_file_name(self.model_type,self.model_version,self.model_alg,self.model_time,'py')
        self.obj_module_orig = None
    
    def __get_pipe_obj_module(self):
        
        pipe = [value for key,value in self.model_info.items() if isinstance(value,Pipeline)][0]
        obj_module =  dict((key,type(val).__module__) for key,val in (pipe.named_steps).items())
        
        return pipe,obj_module
        
    
    def change_pipe_obj_module(self,change_cls_module:list = self_cls_module,module_change = '__main__'):
        
        pipe,obj_module_orig = self.__get_pipe_obj_module()
        
        if self.obj_module_orig is None:
            self.obj_module_orig = obj_module_orig
        
        
        named_steps = pipe.named_steps
        self.change_cls_key = [key for key,val in self.obj_module_orig.items() if val in change_cls_module]
        
        for obj in self.change_cls_key:
            
            cls = named_steps[obj]
            type(cls).__module__ = module_change
        
        
    def reset_pipe_obj_module(self):
        
        (pipe,_) = self.__get_pipe_obj_module()
        named_steps = pipe.named_steps
         
        for obj in self.change_cls_key:
            cls = named_steps[obj]
            type(cls).__module__ = self.obj_module_orig[obj]
            
    
    def model_info_to_pickle(self,path:str = 'model',is_module_change:bool = True,
                             change_cls_module:list = self_cls_module,module_change = '__main__'):
        
        if self.model_info is not None:
            
            if is_module_change:
                
                self.change_pipe_obj_module(change_cls_module,module_change)
                save_pickle_file(os.path.join(path,self.model_file_pickle),self.model_info,False)
                self.reset_pipe_obj_module()
            
            else:
                save_pickle_file(os.path.join(path,self.model_file_pickle),self.model_info,False)

            print('模型文件pickle成功')
        
        else:
            raise ValueError('model_info is None:无模型信息')
    
    
    def get_cls_pack_code(self,cls_module:list = self_cls_module):
        
        (pipe,_) = self.__get_pipe_obj_module()
        named_steps = pipe.named_steps
        
        self.all_cls = []
        
        for key,val in named_steps.items():
            
            res_cls = get_all_cls(type(val),cls_module)
            self.all_cls.extend(res_cls)
            
            
    def get_code_from_cls(self):

        try:
           self.cls_code = get_obj_lst_code(self.all_cls)
        
        except:
            print("需要加载源码的类不明确,先执行get_cls_pack_code方法")
    

    def get_code_from_obj(self,obj_lst:list = public_obj_lst):
        
        self.obj_code = get_obj_lst_code(obj_lst) 
        
        
    def cls_code_to_py(self,path = 'model',cls_module:list = self_cls_module,
                       public_obj:list = public_obj_lst,add_code:list = public_import_code):
        
        self.get_cls_pack_code(cls_module)
        self.get_code_from_cls()
        self.get_code_from_obj(obj_lst = public_obj)
        self.add_code = add_code
        
        all_code = self.add_code + self.obj_code + self.cls_code
        
        
        path_file = os.path.join(path,self.model_file_py)
        with open(path_file,'w',encoding= "utf-8") as file:
            for code in all_code:
                file.write(code)
        
        print('源码生成py文件成功')



if __name__ == '__main__':
    
    pass





