# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2019

@author: xiaox
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
   )

import pandas as pd
import numpy as np
import copy

from src.basic_utils import(
    process_df_fill_null,
    compute_df_null_rate,
    compute_df_median,
    compute_df_mean,
    compute_df_mode,
    compute_df_skew,
    f_df_woe,
    f_var_encode,
    f_var_replace,
    f_var_type_transf
   )

from src.var_bin_utils import (
        f_df_str_chisq_group,
        f_df_num_chisq_group,
        f_df_num_dtree_group,
        f_df_num_quantile_group,
        f_df_num_equal_width_group,
        f_binning_num
        )

from src.utils import select_var_by_type


class FillNull(BaseEstimator,TransformerMixin):
    
    fill_value_fun = {}
    require_var_type = ['int','float','str','bool']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,fill_type = None,fill_thres = 1.0,fill_value = np.nan):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.fill_thres = fill_thres
        self.fill_type = fill_type
        self.fill_value = fill_value
        self.fit_status = False
        
    def fit(self,X,y = None):
        pass
    
    def transform(self,X):
        df = copy.deepcopy(X)
        df = process_df_fill_null(df,self.fill_info)
        
        return df
    
    def fit_transform(self,X,y = None):
        self.fit(X)
        df = self.transform(X)
        
        return df
    
    
class FillNullVarNum(FillNull):
    
    fill_value_fun = {'median': compute_df_median,
                'mean': compute_df_mean}
    
    require_var_type = ['int','float']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,fill_type = 'median',fill_thres = 1.0,fill_value = 0):
        super().__init__(cols_lst,uid,y,fill_type,fill_thres)
        
        
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]
        
        self.null_rate = compute_df_null_rate(X[self.cols_lst])
        self.fill_info = self.fill_value_fun[self.fill_type](X,self.cols_lst)
        for col in self.cols_lst:
            if self.null_rate[col] >= self.fill_thres:
                self.fill_info[col] = np.nan
            
        self.fit_status = True
    
    
class FillNullVarStr(FillNull):
    
    fill_value_fun = {'mode': compute_df_mode }
    require_var_type = ['str']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,fill_type = 'mode',fill_thres = 1,fill_value = 'null'):
        super().__init__(cols_lst,uid,y,fill_type,fill_thres)
        
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]
        
        self.null_rate = compute_df_null_rate(X[self.cols_lst])
        self.fill_info = self.fill_value_fun[self.fill_type](X,self.cols_lst)
        for col in self.cols_lst:
            if self.null_rate[col] >= self.fill_thres:
                self.fill_info[col] = 'null'
    
    
class VarReplace(BaseEstimator,TransformerMixin):
    
    require_var_type = []
    
    def __init__(self,cols_lst,old_val_lst,new_val_lst,uid = None,y = None):
        self.cols_lst = cols_lst
        self.old_val_lst = old_val_lst
        self.new_val_lst = new_val_lst
        self.uid = uid
        self.y = y
        self.fit_status = False
    
    def fit(self,X = None,y = None):
        
        if len(self.cols_lst) == len(self.old_val_lst) and len(self.old_val_lst) == len(self.new_val_lst):
            self.conf = {i[0]:[i[1],i[2]]for i in zip(self.cols_lst,self.old_val_lst,self.new_val_lst)}
            self.fit_status = True
        else:
            raise ValueError('参数cols_lst、old_val_lst和new_val_lst列表长度不一致！')
    
    def transform(self,X,y = None):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = f_var_replace(df[col],self.conf[col][0],self.conf[col][1])
        
        return df
    
    def fit_transform(self,X,y = None):
        
        self.fit()
        df = self.transform(X,y)
        return df    

class TransfVarType(BaseEstimator,TransformerMixin):
    
    require_var_type = []
    
    def __init__(self,cols_lst,transf_type_lst,uid = None,y = None):
        self.cols_lst = cols_lst
        self.transf_type_lst = transf_type_lst
        self.uid = uid
        self.y = y
        self.fit_status = False
    
    def fit(self,X = None,y = None):
        
        if len(self.cols_lst) == len(self.transf_type_lst):
            self.conf = {i[0]:i[1] for i in zip(self.cols_lst,self.transf_type_lst)}
            self.fit_status = True
            
        else:
            raise ValueError("参数cols_lst和transf_type_lst列表长度不一致")
        
    def transform(self,X,y = None):
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = f_var_type_transf(df[col],self.conf[col])
            
        return df
        
    def fit_transform(self,X,y = None):
        
        self.fit()
        df = self.transform(X,y)
        return df
    

class TransfVarEncode(BaseEstimator, TransformerMixin):
    
    requre_var_type = ['str','int','float','bool']
    
    def __init__(self,cols_lst:list,dic_lst:list,default_val = None,enc_type = 'num',uid = None,y = None):
        
        self.cols_lst = cols_lst
        self.dic_lst = dic_lst
        self.default_val = default_val
        self.enc_type = enc_type
        self.uid = uid
        self.y = y
        self.fit_status = False
        
    def fit(self,X = None,y = None):
        
        if len(self.cols_lst) == len(self.dic_lst):
            
            self.conf = {i[0]:i[1] for i in zip(self.cols_lst,self.dic_lst)}
            self.fit_status = True
        else:
            raise ValueError("参数cols_lst和dic_lst列表不一致")
    
    def transform(self,X,y = None):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = f_var_encode(df[col],self.conf[col],self.default_val,self.enc_type)
            
        return df    
        
    def fit_transform(self,X,y = None):
        
        self.fit(self,X)
        df = self.transform(X)
        return df


    
class LabelEnc(BaseEstimator, TransformerMixin):
    
    require_var_type = ['str']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.LabelEncoders = {col:LabelEncoder() for col in cols_lst}
        self.fit_status = False
    
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
            #self.LabelEncoders = {col:LabelEncoder() for col in self.cols_lst}
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]
        
        self.LabelEncoders = {col:LabelEncoder() for col in self.cols_lst}
        
        for col in self.cols_lst:
            self.LabelEncoders[col].fit(X[col])
            
        self.fit_status = True
        
    def transform(self,X):
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = self.LabelEncoders[col].transform(df[col])
        
        return df
    
    def fit_transform(self,X,y = None):
        self.fit(X)
        df = self.transform(X)
        return df
    

class OneHotEnc(BaseEstimator, TransformerMixin):
     
    require_var_type = ['str']
        
    def __init__(self,cols_lst:list = [],uid = None,y = None):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.OneHotEncoders = {col:OneHotEncoder(handle_unknown = 'ignore') for col in cols_lst}
        self.fit_status = False
    
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
            #self.OneHotEncoders = {col:OneHotEncoder(handle_unknown = 'ignore') for col in self.cols_lst}
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]
        
        self.OneHotEncoders = {col:OneHotEncoder(handle_unknown = 'ignore') for col in self.cols_lst}
                   
        for col in self.cols_lst:
            self.OneHotEncoders[col].fit(X[col].values.reshape(-1,1))
        self.fit_status = True
    
    def transform(self,X):
        df = copy.deepcopy(X)
        df.reset_index(drop = True,inplace =  True)
        df.drop(columns = self.cols_lst,inplace = True)
        for col in self.cols_lst:
            df_tmp = self.OneHotEncoders[col].transform(X[col].values.reshape(-1,1)).toarray()
            col_name = [col + '_' + cat for cat in self.OneHotEncoders[col].categories_[0]]
            df_tmp = pd.DataFrame(df_tmp,columns = col_name)
            df = df.join(df_tmp)
            
        return df
    
    def fit_transform(self,X,y = None):
        self.fit(X)
        df = self.transform(X)
        
        return df    
    
class MathTransfVarNum(BaseEstimator,TransformerMixin):
    
    require_var_type = ['int','float']
    transf_fun = {'log1p': np.log1p,
                  'sqrt': np.sqrt}
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,skew = 0.5,transf = 'log1p'):
        self.cols_lst = cols_lst
        self.uid = uid 
        self.y = y
        self.skew = skew
        self.transf = transf
        self.fit_status = False
    
    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]        
        
        self.skew_info = compute_df_skew(X[self.cols_lst])
        self.status = True
    
    def transform(self,X):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            if abs(self.skew_info[col]) > self.skew:
                df[col] = self.transf_fun[self.transf](df[col])
                
        return df
    
    def fit_transform(self,X,y = None):
        
        self.fit(X)
        df = self.transform(X)
        return df


class StandardScalerVarNum(BaseEstimator,TransformerMixin):
    
    require_var_type = ['int','float']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,with_mean = True,with_std = True):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.with_mean = with_mean
        self.with_std = with_std
        self.StandardScalers = {col:StandardScaler(copy = False,with_mean = self.with_mean,with_std = self.with_std) for col in self.cols_lst}
        self.status = False
        
        
    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]  
           
        self.StandardScalers = {col:StandardScaler(copy = False,with_mean = self.with_mean,with_std = self.with_std) for col in self.cols_lst}
        
        for col in self.cols_lst:
            self.StandardScalers[col].fit(X[col].values.reshape(-1,1))
        
        self.fit_status = True
    
    
    def transform(self,X):
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = self.StandardScalers[col].transform(df[col].values.reshape(-1,1))
            
        return df
    
    
    def fit_transform(self,X,y = None):
        
        self.fit(X)
        df = self.transform(X)
        return df    

class MinMaxScalerVarNum(BaseEstimator,TransformerMixin):
    
    require_var_type = ['int','float']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,feature_range = (0, 1)):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.feature_range = feature_range
        self.MinMaxScalers = {col:MinMaxScaler(feature_range = self.feature_range) for col in self.cols_lst}
        self.status = False
        
    def fit(self,X,y = None):
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]  
           
        self.MinMaxScalers = {col:MinMaxScaler(feature_range = self.feature_range) for col in self.cols_lst}
        
        for col in self.cols_lst:
            self.MinMaxScalers[col].fit(X[col].values.reshape(-1,1))
        
        self.fit_status = True
        
    def transform(self,X):
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = self.MinMaxScalers[col].transform(df[col].values.reshape(-1,1))
            
        return df
    
    def fit_transform(self,X,y = None):
        
        self.fit(X)
        df = self.transform(X)
        return df    
    

class BinVarStrChisq(BaseEstimator,TransformerMixin):
    
    require_var_type = ['str']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,p_value = 0.05,
                 pct = 0.05,max_groups = 10,num_least = 30):
        
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.p_value = p_value
        self.pct = pct
        self.max_groups = max_groups
        self.num_least = num_least
        self.fit_status = False
        
    
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for  col in self.cols_lst if col in X.columns.tolist()]
    
        
        self.bin_info = f_df_str_chisq_group(X = X[self.cols_lst],
                                             y = y,
                                             p_value = self.p_value,
                                             pct = self.pct,
                                             max_groups = self.max_groups,
                                             num_least = self.num_least
                                             )
        
        self.fit_status = True
    
    
    def transform(self,X):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            
            default_val = list(self.bin_info[col].values())[0]
            df[col] = df[col].apply(lambda x: 'null' if pd.isnull(x) else self.bin_info[col].get(x,default_val))
        
        return df
    
    
    def fit_transform(self,X,y = None):
        
        self.fit(X,y)
        df = self.transform(X)
        
        return df
    

class BinVarNum(BaseEstimator,TransformerMixin):
    
    require_var_type = ['int','float']
    bin_fun = {'dtree':f_df_num_dtree_group,
               'chisq':f_df_num_chisq_group,
               'quantile':f_df_num_quantile_group,
               'equal_width':f_df_num_equal_width_group}
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,bin_type = 'dtree',
                 p_value = 0.05,pct = 0.05,max_groups = 10,num_least = 20,
                 criterion = 'gini',min_samples_leaf = 0.05,max_leaf_nodes = 10, min_impurity_decrease = 1e-5,
                 q = 5):
        
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.bin_type = bin_type
        self.p_value = p_value
        self.pct = pct
        self.max_groups = max_groups
        self.num_least = num_least
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.q = q
        self.cut_info = {}
        self.fit_status = False 
        
    def fit(self,X,y = None):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = [col for col in select_var_by_type(X,uid = self.uid,y = self.y,var_type = ['int','float'])]
        
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]
            

        if self.bin_type == 'dtree':
            self.cut_info = self.bin_fun[self.bin_type](X[self.cols_lst],y = y,
                             criterion = self.criterion,min_samples_leaf = self.min_samples_leaf,
                             max_leaf_nodes = self.max_leaf_nodes, min_impurity_decrease = self.min_impurity_decrease)
                        
        elif self.bin_type == 'chisq':
            self.cut_info = self.bin_fun[self.bin_type](X[self.cols_lst],y = y,
                             p_value = self.p_value,pct = self.pct,max_groups = self.max_groups,num_least = self.num_least)
        
        
        elif self.bin_type in ['quantile','equal_width']:
            self.cut_info = self.bin_fun[self.bin_type](X[self.cols_lst],self.q)
            
        
        else:
            raise ValueError('bin_type参数只支持dtree、chisq、quantile和equal_width值')
        self.fit_status = True
    
    
    def transform(self,X):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            if len(self.cut_info[col]) != 0:
                df[col] = f_binning_num(df[col],self.cut_info[col])
        
        return df
                 
                
    def fit_transform(self,X,y = None):
        
        self.fit(X,y)
        df = self.transform(X)
        return df


class WoeVarStr(BaseEstimator,TransformerMixin):
    
    require_var_type = ['str']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.fit_status = False
    
    def fit(self,X,y):
        
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns]
        
        self.woe_info = f_df_woe(X[self.cols_lst],y)
        
        self.fit_status = True
        
    def transform(self,X):
        
        df = copy.deepcopy(X)
        for col in self.cols_lst:
            df[col] = df[col].apply(lambda x:self.woe_info[col].get(x,np.nan))
            
        return df
    
    def fit_transform(self,X,y):
        
        self.fit(X,y)
        df = self.transform(X)
        return df


