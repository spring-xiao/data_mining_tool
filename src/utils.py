# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2020

@author: xiaox
"""


from src.basic_utils import get_vars_type
from sklearn.metrics import roc_auc_score,roc_curve


def select_var_by_type(df,uid = None,y = None,var_type = ['int','float']):
    
    if uid is None:
        uid = []
    elif not isinstance(uid,list):
        uid = [uid]
    
    if y is None:
        y = []
    elif not isinstance(y,list):
        y = [y]
    
    if not isinstance(var_type,list):
        var_type = [var_type]
    
    f_types = get_vars_type(df)
    
    if var_type[0] == 'all':
        cols_select = list(f_types.keys())
    else:
        cols_select = [k for k,v in f_types.items() if v in var_type]
    
    cols_select = list(set(cols_select) - set(uid + y))
    
    return cols_select


def get_ks(y_true,y_score):
    
    fpr, tpr, thresholds = roc_curve(y_true = y_true, y_score = y_score)
    ks = max(abs(tpr - fpr))
    return ks
