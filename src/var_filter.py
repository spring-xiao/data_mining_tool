# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2019

@author: xiaox
"""


from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np
import copy

from src.basic_utils import(
    compute_df_nunique_cnt,
    compute_df_null_rate,
    compute_df_median,
    compute_df_mean,
    compute_df_mode,
    computer_df_importance_xgb_regressor,
    compute_df_corr,
    f_df_iv
   )
from src.utils import select_var_by_type

class VarFilter(BaseEstimator,TransformerMixin):
    
    def __init__(self,cols_lst:list = [],uid = None,y = None):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.var_filter = {}
        self.fit_status = False
        
    def fit(self,X,y = None):
        pass
    
    def transform(self,X):
           
        df = copy.deepcopy(X)
        cols_except = list(self.var_filter.keys())
        df.drop(columns = cols_except,inplace = True)
        return df
    
    def fit_transform(self,X,y = None):
        
        self.fit(X,y)
        df = self.transform(X)
        return df

class VarFilterNull(VarFilter):
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,filter_thres = 0.9):
        super().__init__(cols_lst,uid,y)
        self.filter_thres = filter_thres
 
    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst =  select_var_by_type(X,uid = self.uid,y = self.y,var_type = 'all')
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        
            
        self.null_rate = compute_df_null_rate(X[self.cols_lst])
        self.var_filter = {k:v for k,v in self.null_rate.items() if v >= self.filter_thres}
        self.fit_status = True
    
#     def transform(self,X):
        
#         df = copy.deepcopy(X)
#         cols_except = list(self.var_filter.keys())
#         df.drop(columns = cols_except,inplace = True)
#         return df
    
class VarFilterUniqueSingle(VarFilter):
    '删除唯一值'
    
    def __init__(self,cols_lst:list = [],uid = None,y = None):
        super().__init__(cols_lst,uid,y)
        self.uid = uid
        self.y = y
        self.filter_thres = 1

    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = 'str')
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        

        self.nunique_cnt = compute_df_nunique_cnt(X[self.cols_lst])
        self.var_filter = {k:v for k,v in self.nunique_cnt.items() if v == self.filter_thres}
        self.fit_status = True
    
#     def transform(self,X):
        
#         df = copy.deepcopy(X)
#         cols_except = list(self.var_filter.keys())
#         df.drop(columns = cols_except,inplace = True)
#         return df

class VarFilterUniqueMulti(VarFilter):
    '删除字符串值种类数过多'
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,filter_thres = 50):
        super().__init__(cols_lst,uid,y)
        self.filter_thres = filter_thres
        
    def fit(self,X,y = None):
        if len(self.cols_lst) ==0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = 'str')
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        

        self.nunique_cnt = compute_df_nunique_cnt(X[self.cols_lst])
        self.var_filter = {k:v for k,v in self.nunique_cnt.items() if v >= self.filter_thres}
        self.fit_status = True
    
#     def transform(self,X):
        
#         df = copy.deepcopy(X)
#         cols_except = list(self.var_filter.keys())
#         df.drop(columns = cols_except,inplace = True)
#         return df
 
    
class VarFilterImpXgbRegressor(VarFilter):
    
    require_var_type = ['int','float','bool']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,filter_thres = 0):
        super().__init__(cols_lst,uid,y)
        self.filter_thres = filter_thres
        
    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst =  select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        
        
        if y is None:
            y = X[self.y]
        self.var_importance = computer_df_importance_xgb_regressor(X[self.cols_lst],y)
        self.var_filter = {k:v for k,v in self.var_importance.items() if v <= self.filter_thres}
        self.fit_status = True
        
class VarFilterCorr(VarFilter):
    
    require_var_type = ['int','float']

    def __init__(self,cols_lst:list = [],uid = None,y = None,filter_thres = 0.05):
        super().__init__(cols_lst,uid,y)
        self.filter_thres = filter_thres

    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        
        
        if y is None:
            y = X[self.y]
            
        self.corr_info = compute_df_corr(X[self.cols_lst],y)
        self.var_filter = {k:v for k,v in self.corr_info.items() if abs(v) < self.filter_thres}
        self.fit_status = True 
        
        
class VarFilterIv(VarFilter):
    
    require_var_type = ['str']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,iv_thres = 0.02):
        super().__init__(cols_lst,uid,y)
        self.iv_thres = iv_thres
        
        
    def fit(self,X,y):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
            
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.to_list()]
    
        self.iv_info = f_df_iv(X[self.cols_lst],y)
        self.var_filter = {k:v for k,v in self.iv_info.items() if v < self.iv_thres}
        self.fit_status = True       
        
        
        
               