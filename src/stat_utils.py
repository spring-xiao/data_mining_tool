# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:33:45 2020

@author: xiaox
"""


import copy
import numpy as np
import pandas as pd


def get_df_iv_info(X,y):
    
    df_info = pd.DataFrame()
    for col in X.columns:
        
        df_tmp = get_var_iv_info(X[col],y)
        cols = df_tmp.columns.tolist()
        df_tmp['var'] = col
        df_tmp = df_tmp[['var'] + cols]
        
        df_info = df_info.append(df_tmp)
        
    df_info.reset_index(drop = True,inplace = True)    
    
    return df_info


def get_var_iv_info(x,y):
    
    x_var = copy.deepcopy(x)
    y_var = copy.deepcopy(y)
    
    x_var = x_var.astype('str')
    x_var.reset_index(drop = True,inplace = True)
    y_var.reset_index(drop = True,inplace = True)
    
    df_tmp = pd.crosstab(x_var,y_var)
    df_tmp.rename(columns = {0:'good_cnt',1:'bad_cnt','0':'good_cnt','1':'bad_cnt'},inplace = True)
   
    df_tmp['good_pct'] = df_tmp['good_cnt']/df_tmp['good_cnt'].sum()
    df_tmp['bad_pct'] = df_tmp['bad_cnt']/df_tmp['bad_cnt'].sum()
     
    df_tmp['woe'] = np.log(df_tmp['bad_pct']/df_tmp['good_pct'])
    df_tmp['miv'] = (df_tmp['bad_pct'] - df_tmp['good_pct'])*df_tmp['woe'].replace([-np.inf,np.inf],0)
    df_tmp['iv'] = df_tmp['miv'].sum()
    
    df_tmp['bad_rate'] = df_tmp['bad_cnt']/(df_tmp['bad_cnt'] +df_tmp['good_cnt'])
    df_tmp['pct'] = (df_tmp['good_cnt'] + df_tmp['bad_cnt'])/(df_tmp['good_cnt'].sum() + df_tmp['bad_cnt'].sum())
    
    #df_tmp.reset_index(drop = False,inplace = True)
    df_tmp['var_value'] = list(df_tmp.index)

    cols = ['var_value','good_cnt','bad_cnt','good_pct','bad_pct',
            'woe','miv','iv','bad_rate','pct']
    
    df_tmp = df_tmp[cols]
    

    return df_tmp






