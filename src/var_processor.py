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
    f_var_encode
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
    
    
class TransfVarEncode(BaseEstimator, TransformerMixin):
    
    requre_var_type = ['str','int','float','bool']
    
    def __init__(self,cols_lst:list,dic_lst:list,default_val = None,uid = None,y = None):
        
        self.cols_lst = cols_lst
        self.dic_lst = dic_lst
        self.default_val = default_val
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
            df[col] = f_var_encode(df[col],self.conf[col],self.default_val)
            
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
    

