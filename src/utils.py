# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2020

@author: xiaox
"""

import numpy as np

def transf_data_type(data:dict):
    """
    numpy 的数据类型转换为python数据类型
    
    """
    res = {}
    for key,val in data.items():
        
        if isinstance(val,np.str):
            val = str(val)
        elif isinstance(val,(np.float,np.float16,np.float32,np.float64)):
            val = float(val)
        elif isinstance(val,(np.int,np.int16,np.int32,np.int32,np.int64)):
            val = int(val)
            
        res.update({key:val})
        
    return res


#******************************************************************************
#import inspect
#
#utils_fun = dict((key,val) for key,val in locals().items() \
#                if inspect.isfunction(val) and val.__module__ in( '__main__','src.utils'))




