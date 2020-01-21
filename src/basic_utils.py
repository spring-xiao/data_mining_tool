import pandas as pd
import numpy as np
import re
from scipy.stats import skew


def compute_df_skew(df):
    
    skew_info = {}
    for col in df.columns.tolist():
        skew_info[col] = compute_var_skew(df[col])
        
    return skew_info
    
def compute_var_skew(data):
    
    data = data[data.notnull()]
    skewness = skew(data)
    
    return skewness

def process_df_fill_null(df,fill_info:dict):
    
    for k,v in fill_info.items():
        df[k] = process_var_fill_null(data = df[k],fill_value = v)

    return df

def process_var_fill_null(data,fill_value):
    
    data.replace(np.nan,fill_value,inplace = True)
    return data


def compute_df_mean(df,cols_num = None):
    
    if cols_num is None:
        cols_num = df.columns.tolist()
    
    cols_mean = {}
    for col in cols_num:
        cols_mean[col] = compute_var_mean(df[col])
    
    return cols_mean
        
def compute_df_median(df,cols_num = None):
    
    if cols_num is None:
        cols_num = df.columns.tolist()
    
    cols_medain = {}
    for col in cols_num:
        cols_medain[col] = compute_var_median(df[col])
    
    return cols_medain

def compute_df_mode(df,cols_str = None):
    
    if cols_str is None:
        cols_str = df.columns.tolist()
        
    cols_mode = {}
    for col in cols_str:
        cols_mode[col] = compute_var_mode(df[col])
        
    return cols_mode

def compute_var_mean(x):

    mean = x.mean()
    return mean

def compute_var_median(x):

    median = x.median()    
    return median

def compute_var_mode(x):
    
    category = dict(x.groupby(x).count())
    mode = max(category,key = category.get)
    
    return mode


def get_vars_type(df):
    
    vars_type = dict(df.dtypes.apply(lambda x:str(x)))
    f_types = {}
    for k,v in vars_type.items():
        if len(re.findall('int',v)) > 0:
            f_type = 'int'
        elif len(re.findall('float',v)) > 0:
            f_type = 'float'
        elif len(re.findall('object',v)) > 0:
            f_type = 'str'
        elif len(re.findall('bool',v)) > 0:
            f_type = 'bool'
        elif len(re.findall('datetime',v)) > 0:
            f_type = 'time'
        else:
            f_type = 'unknown'
            
        f_types[k] = f_type
        
    return f_types

def compute_df_null_rate(df):
    
    cols = df.columns.tolist()
    null_rate = {}
    for col in cols:
        null_rate[col] = compute_var_null_rate(df[col])
    
    return null_rate

def compute_df_null_cnt(df):
    
    cols = df.columns.tolist()
    null_cnt = {}
    for col in cols:
        null_cnt[col] = compute_var_null_cnt(df[col])
    
    return null_cnt

def compute_var_null_cnt(x):
    
    null_cnt = x.isnull().sum()
    
    return null_cnt

def compute_var_null_rate(x):
    
    null_cnt = compute_var_null_cnt(x)
    all_cnt = len(x)
    null_rate = round(null_cnt/all_cnt,6)
    
    return null_rate


def compute_df_nunique_cnt(df):
 
    nunique_cnt = {}
    for col in df.columns:
        nunique_cnt[col] = compute_var_nunique_cnt(df[col])
    
    return nunique_cnt


def compute_df_nunique_rate(df):

    nunique_rate = {}
    for col in df.columns:
        nunique_rate[col] = compute_var_nunique_rate(df[col])
    
    return nunique_rate    

def compute_var_nunique_cnt(x):

    nunique_cnt = x.nunique()
    return nunique_cnt

def compute_var_nunique_rate(x):

    nunique_rate = round(compute_var_nunique_cnt(x)/len(x),6)
    
    return nunique_rate
    

def computer_df_importance_xgb_regressor(X,y):
 
    from operator import itemgetter
    from xgboost import XGBRegressor
    import sys
    from warnings import filterwarnings
    filterwarnings('ignore')
    
    learning_rate_list = [0.05,0.1,0.15,0.2,0.25]
    subsample_list = [0.8,0.85,0.9,0.95,1.0]
    colsample_bytree_list = [0.8,0.85,0.9,0.95,1.0]
    
    epoch = 0
    iters = 6
    cnt =min(len(learning_rate_list),len(subsample_list),len(colsample_bytree_list))
    sys.stdout.write("\rxgb_regressor变量筛选完成... finished:{}%    \r".format(round(epoch*100/(iters*cnt),2)))
    var_importance = []
    for i in range(iters):
        
        np.random.shuffle(learning_rate_list)
        np.random.shuffle(subsample_list)
        np.random.shuffle(colsample_bytree_list)
        zip_ = zip(learning_rate_list,subsample_list,colsample_bytree_list)

        for learning_rate,subsample,colsample_bytree in zip_:
            epoch+=1
            model_xgb = XGBRegressor(max_depth = 3,
                                     learning_rate = learning_rate,
                                     n_estimamters = 1000,
                                     booster ='gbtree',
                                     objective = 'reg:squarederror',
                                     subsample = subsample,
                                     colsample_bytree = colsample_bytree,
                                     #colsample_bylevel = 1,
                                     importance_type = 'gain',
                                     reg_alpha = 0.1,
                                     reg_lambda = 0.1
                                    )

            model_xgb.fit(X = X,y = y)
            var_importance.append(model_xgb.feature_importances_)
            sys.stdout.write("\rxgb_regressor变量筛选完成... finished:{}%    \r".format(round(epoch*100/(iters*cnt),2)))
    
    sys.stdout.write("\rxgb_regressor变量筛选完成... finished:{}%    \r".format(round(epoch*100/(iters*cnt),2)))
    var_importance = np.array(var_importance).mean(axis=0)
    var_importance = {key:value for key,value in zip(X.columns.tolist(),var_importance)}
    var_importance = sorted(var_importance.items(),key = itemgetter(1),reverse = True)
    var_importance = {i[0]:i[1] for i in var_importance}
    
    filterwarnings('default')
    
    return var_importance


def compute_df_corr(X,y,method = 'pearson'):
    
    corr_info = {}
    for col in X.columns:
        corr_info[col] = compute_var_corr(X[col],y,method = method)
        
    return corr_info

def compute_var_corr(x,y,method = 'pearson'):
    
    corr = y.corr(x,method = method)
    
    return corr


def f_encode(val,dic,default_val = None,enc_type = 'num'):
    
    if enc_type == 'num':
        res = dic.get(val,default_val)
    elif enc_type == 'str':
        res = dic.get(val,default_val)
        if res is not None:
            res = 'bin'+str(res)
    else:
        raise ValueError("enc_type参数要求为:'num'或'str'")
    
    return  res

def f_var_encode(X,dic,default_var = None,enc_type = 'num'):
    
    res = X.apply(lambda x:f_encode(x,dic,default_var,enc_type))
    return res


def f_var_replace(X,old_val,new_val):
    
    res = X.apply(lambda x:new_val if x == old_val else x)
    return res


def f_var_type_transf(X,transf_type):
    
    if transf_type in ['str','int','float','bool']:
        if transf_type ==  'str':
            res = X.apply(lambda x: np.nan if pd.isnull(x) else str(x))
        elif transf_type ==  'int':
            res = X.apply(lambda x: np.nan if pd.isnull(x) else int(x))
        elif transf_type ==  'float':
            res = X.apply(lambda x: np.nan if pd.isnull(x) else float(x)) 
        elif transf_type ==  'bool':
            res = X.apply(lambda x: np.nan if pd.isnull(x) else bool(x)) 
            
    else:
        raise ValueError("参数transf_type必须在'str','int','float','bool'中取值")
    
    return res
