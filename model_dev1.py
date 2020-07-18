# -*- coding: utf-8 -*-

"""
Created on Tue Jan 21 09:33:31 2020

@author: xiaoxiang
"""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils import get_vars_type 
from src.var_filter import VarFilterUniqueMulti
from src.var_processor import (
        OneHotEnc,
        FillNullVarNum,
        FillNullVarStr
        )


from xgboost import XGBClassifier
import time

#os.chdir('E:\spyder\data-mining-tool')


accepts = pd.read_csv('data/accepts.csv')

'''
'application_id',
'account_number',
'bad_ind',
'vehicle_year',
'vehicle_make',
'bankruptcy_ind',
'tot_derog',
'tot_tr',
'age_oldest_tr',
'tot_open_tr',
'tot_rev_tr',
'tot_rev_debt',
'tot_rev_line',
'rev_util',
'fico_score',
'purch_price',
'msrp',
'down_pyt',
'loan_term',
'loan_amt',
'ltv',
'tot_income',
'veh_mileage',
'used_ind'
'''

cols = accepts.columns.tolist()
uid = ['account_number','application_id']
y_var = 'bad_ind'
x_var = list(set(cols) - set(uid + [y_var]))

#x_var_str =
#x_var_num = 

#get_vars_type(accepts)

X_train,X_test,y_train,y_test = train_test_split(
        accepts[x_var],accepts[y_var],test_size = 0.33,random_state = 42)


# =============================================================================
# var_str_fillnull = FillNullVarStr(cols_lst = [],y = y_var,uid = uid,fill_thres = 0)
# var_str_fillnull.fit(X = X_train,y = y_train)
# 
# X_train = var_str_fillnull.transform(X = X_train)
# X_test = var_str_fillnull.transform(X = X_test)
# 
# var_onehot = OneHotEnc(cols_lst = [],y = y_var,uid = uid)
# var_onehot.fit(X = X_train,y = y_train)
# =============================================================================

#******************特征处理流***********************
var_filter_multi = VarFilterUniqueMulti(cols_lst = [],uid = uid,y = y_var,filter_thres = 15)
var_str_fillnull = FillNullVarStr(cols_lst = [],y = y_var,uid = uid,fill_thres = 0)
var_onehot = OneHotEnc(cols_lst = [],y = y_var,uid = uid)


vars_processor = [('var_filter_multi',var_filter_multi),
                  ('var_str_fillnull',var_str_fillnull),
                  ('var_onehot',var_onehot)]

var_pipe = Pipeline(steps = vars_processor)
var_pipe.fit(X = X_train)


X_train = var_pipe.transform(X = X_train)
X_test = var_pipe.transform(X = X_test)


#*******************模型建立************************

params = {'max_depth':2,
          'learning_rate': 0.001,
          'booster': 'gbtree',
          'reg_alpha' : 0.8,
          'reg_lambda' : 0.9,
          'objective' : 'binary:logistic'
          }  

xgb_model = XGBClassifier(**params)
xgb_model.fit(X = X_train,y = y_train)

xgb_model.predict_proba(X_train.iloc[0:5])[:,1]


#*************************************************
model_info = {'x_var':x_var,
              'var_pipe':var_pipe,
              'model':xgb_model
             }

#******************************************************************************
#模型打包
from src.model_pack import ModelPack

model_pack = ModelPack(model_info = model_info)
model_pack.model_info_to_pickle()
model_pack.cls_code_to_py()








