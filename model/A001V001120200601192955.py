from scipy.stats import chi2,skew
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.tree import  DecisionTreeClassifier as DTree
from sklearn.preprocessing import (
LabelEncoder,
OneHotEncoder,
StandardScaler,
MinMaxScaler)
import pandas as pd
import numpy as np
import re
import copy
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
def f_var_woe(x,y):
    
    
    x.reset_index(drop = True,inplace = True)
    y.reset_index(drop = True,inplace = True)
    x = x.astype(str)
    df_tmp = pd.crosstab(x,y)
    df_tmp.rename(columns = {0:'good_cnt',1:'bad_cnt','0':'good_cnt','1':'bad_cnt'},inplace = True)
    df_tmp['good_pct'] = df_tmp['good_cnt']/df_tmp['good_cnt'].sum()
    df_tmp['bad_pct'] = df_tmp['bad_cnt']/df_tmp['bad_cnt'].sum()
    df_tmp['woe'] = round(np.log(df_tmp['bad_pct']/(df_tmp['good_pct'] + 1e-8)),6)

    woe = df_tmp['woe'].to_dict()
    
    return woe
def f_df_woe(X,y):
    
    woe_info = {}
    for col in X.columns:
        woe_info[col] = f_var_woe(X[col],y)
        
    return woe_info
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
def f_var_iv(x,y):
    
    
    x.reset_index(drop = True,inplace = True)
    y.reset_index(drop = True,inplace = True)
    x = x.astype(str)
    df_tmp = pd.crosstab(x,y)
    df_tmp.rename(columns = {0:'good_cnt',1:'bad_cnt','0':'good_cnt','1':'bad_cnt'},inplace = True)
    df_tmp['good_pct'] = df_tmp['good_cnt']/df_tmp['good_cnt'].sum()
    df_tmp['bad_pct'] = df_tmp['bad_cnt']/df_tmp['bad_cnt'].sum()
    
    df_tmp['woe'] = (df_tmp['bad_pct']/df_tmp['good_pct']).replace(np.inf,0).apply(lambda x:np.log(x) if x !=0 else 0)
    df_tmp['miv'] = round((df_tmp['bad_pct'] - df_tmp['good_pct']) * df_tmp['woe'],6)

    iv = df_tmp['miv'].sum()
    
    return iv
def f_df_iv(X,y):
    
    iv_info = {}
    for col in X.columns:
        iv_info[col] = f_var_iv(X[col],y)
        
    return iv_info
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
def f_var_str_chisq_group(x,y,
                          p_value = 0.05,
                          pct = 0.05,
                          max_groups = 10,
                          num_least = 30):
    """
    
    
    """
    # filterwarnings('ignore')
    df_tmp = pd.concat((x,y),axis = 1)
    col ='x_var'
    y_var = 'y_var'
    df_tmp.columns = [col,y_var]
    df_tmp = df_tmp[df_tmp[col].notnull()]
    df_tmp[col]=df_tmp[col].astype(str)
    static = df_tmp.groupby(by = col)[y_var].agg(['count','sum'])
    static.columns = ['allcnt','badcnt']
    static.reset_index(drop = False,inplace = True)
    static['goodcnt'] = static['allcnt']-static['badcnt']
    static['badrate'] = static['badcnt']/static['allcnt']
    static['pct'] = static['allcnt']/static['allcnt'].sum()
    static.sort_values(by = ['pct'],inplace = True)
    static[col] = static[col].apply(lambda x:[x])
    static = static[[col,'badcnt','goodcnt','pct','allcnt','badrate']]
    np_regroup = np.array(static)

    idx = np_regroup[:,4] <= num_least
    if idx.sum() > 0:
        np_tmp1 = np_regroup[idx,:]
        np_tmp1[0,0] = np_tmp1[:,0].sum()
        np_tmp1[0,1] = np_tmp1[:,1].sum()
        np_tmp1[0,2] = np_tmp1[:,2].sum()
        np_tmp1[0,3] = np_tmp1[:,3].sum()
        np_tmp1[0,4] = np_tmp1[:,4].sum()
        np_tmp1[0,5] = np_tmp1[:,1].sum()/np_tmp1[:,4].sum()

        np_tmp2=np_regroup[np.logical_not(idx),:]

        np_regroup=np.concatenate([np_tmp1[0,:].reshape(1,6),np_tmp2],axis=0)

    np_regroup=np_regroup[np_regroup[:,-1].argsort(),:]
    np_regroup=np_regroup[:,[0,1,2,3]]

    i  =  0
    while (i <=  np_regroup.shape[0] - 1):
        if np_regroup.shape[0] < 2:
            break
        n_sample = np_regroup.shape[0]
        if ((np_regroup[i, 1]  <  num_least) or (np_regroup[i, 2]  <  num_least)):
            if i == n_sample-1:
                np_regroup[i-1, [1,2,3]]  =  np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0]  =  np_regroup[i-1,0]+np_regroup[i, 0]
                np_regroup  =  np.delete(np_regroup, i, 0)
            else:
                np_regroup[i, [1,2,3]]  =  np_regroup[i, [1,2,3]] + np_regroup[i + 1, [1,2,3]]  
                np_regroup[i, 0]  =  np_regroup[i, 0]+np_regroup[i + 1, 0]
                np_regroup  =  np.delete(np_regroup, i + 1, 0)
            i  =  i - 1
        i  =  i + 1

    chi_threshold = round(chi2.isf(q = p_value,df = 1),3)
    chisqList = []
    for i in range(0,np_regroup.shape[0]-1):

        a = np_regroup[i,1]
        c = np_regroup[i,2]
        b = np_regroup[i+1,1]
        d = np_regroup[i+1,2]
        chi = (a*d-b*c)**2*(a+b+c+d)/((a+c)*(b+d)*(a+b)*(c+d)+1e-8)
        chi = round(chi,3)
        chisqList.append(chi)

    chi_threshold_flag = (chisqList < chi_threshold).sum()>0 or np_regroup.shape[0]>max_groups
    while chi_threshold_flag:

        min_index  =  chisqList.index(min(chisqList))
        merge_index = min_index+1
        np_regroup[min_index,[1,2,3]] = np_regroup[min_index,[1,2,3]]+np_regroup[merge_index,[1,2,3]]
        np_regroup[min_index,0] = np_regroup[min_index,0]+np_regroup[merge_index,0]
        np_regroup = np.delete(np_regroup,merge_index,0)
        if np_regroup.shape[0] == 1:
            break

        chisqList = []
        for i in range(0,np_regroup.shape[0]-1):
            a = np_regroup[i,1]
            c = np_regroup[i,2]
            b = np_regroup[i+1,1]
            d = np_regroup[i+1,2]
            chi = (a*d-b*c)**2*(a+b+c+d)/((a+c)*(b+d)*(a+b)*(c+d)+1e-8)
            chi = round(chi,3)
            chisqList.append(chi)
        chi_threshold_flag = (chisqList < chi_threshold).sum()>0 or np_regroup.shape[0]>max_groups

    i  =  0
    while (i <=  np_regroup.shape[0] - 1):
        n_sample = np_regroup.shape[0]
        if n_sample < 2:
            break
        if np_regroup[i, 3]<pct:
            if i == 0:
                np_regroup[i+1,[1,2,3]] = np_regroup[i, [1,2,3]] + np_regroup[i + 1, [1,2,3]]
                np_regroup[i+1,0] = np_regroup[i, 0] + np_regroup[i + 1, 0]
            elif i == n_sample-1:
                np_regroup[i-1, [1,2,3]] = np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0] = np_regroup[i-1, 0]+np_regroup[i, 0]
            elif np_regroup[i-1, 3]>= np_regroup[i+1, 3]:
                np_regroup[i+1, [1,2,3]] = np_regroup[i, [1,2,3]] + np_regroup[i+1, [1,2,3]]
                np_regroup[i+1, 0] = np_regroup[i, 0] + np_regroup[i+1, 0]
            elif np_regroup[i-1, 3]<np_regroup[i+1, 3]:
                np_regroup[i-1, [1,2,3]] = np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0] = np_regroup[i-1, 0]+np_regroup[i, 0]
            np_regroup = np.delete(np_regroup,i,0)
            i  =  i - 1
        i  =  i + 1

    group_dict={}

#    i = 0
#     for group in np_regroup[:,0]:
#         i += 1
#         print(group)
#         group_dict['bin'+str(i)] = list(group)
    
    i = 0
    for groups in np_regroup[:,0]:
        i += 1
        for group in groups:
            group_dict[group] = 'bin'+str(i)

    return group_dict
def f_df_str_chisq_group(X,y,
                         p_value = 0.05,
                         pct = 0.05,
                         max_groups = 10,
                         num_least = 30):
    
    
    bin_info = {}
    for col in X.columns:
        group_dict = f_var_str_chisq_group(x = X[col],
                                           y = y,
                                           p_value = p_value,
                                           pct = pct,
                                           max_groups = max_groups,
                                           num_least = num_least)
        bin_info[col] = group_dict
        
    return bin_info
def f_var_num_chisq_group(x,y,
                          p_value = 0.05,
                          pct = 0.05,
                          max_groups = 10,
                          num_least = 10):
    """
    
    
    """
    df_tmp = pd.concat((x,y),axis = 1)
    col ='x_var'
    y_var = 'y_var'
    df_tmp.columns = [col,y_var]
    df_tmp.sort_values(by = [col],inplace = True)
    static = df_tmp.groupby([col])[y_var].agg(['count','sum'])
    static.columns = ['allcnt','badcnt']
    static.reset_index(drop = False,inplace = True)
    static['goodcnt'] = static['allcnt']-static['badcnt']
    static['pct'] = static['allcnt']/static['allcnt'].sum()  

    np_regroup = static[[col,'badcnt','goodcnt','pct']]
    np_regroup = np.array(np_regroup)
    chi_threshold = round(chi2.isf(q = p_value,df = 1),3)
    
    i  =  0
    while (i <=  np_regroup.shape[0] - 1):
        if np_regroup.shape[0]<2:
            break
        n_sample = np_regroup.shape[0]
        if ((np_regroup[i, 1]  <=  num_least) or (np_regroup[i, 2]  <=  num_least)):
            if i == n_sample-1:
                np_regroup[i-1, [1,2,3]]  =  np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0]  =  np_regroup[i, 0]
                np_regroup  =  np.delete(np_regroup, i, 0)
            else:
                np_regroup[i, [1,2,3]]  =  np_regroup[i, [1,2,3]] + np_regroup[i + 1, [1,2,3]]  
                np_regroup[i, 0]  =  np_regroup[i + 1, 0]
                np_regroup  =  np.delete(np_regroup, i + 1, 0)
            i  =  i - 1
        i  =  i + 1

    chisqList = []
    for i in range(0,np_regroup.shape[0]-1):

        a = np_regroup[i,1]
        c = np_regroup[i,2]
        b = np_regroup[i+1,1]
        d = np_regroup[i+1,2]
        chi = (a*d-b*c)**2*(a+b+c+d)/((a+c)*(b+d)*(a+b)*(c+d)+1e-8)
        chi = round(chi,3)
        chisqList.append(chi)

    chi_threshold_flag = (chisqList < chi_threshold).sum()>0 or np_regroup.shape[0]>max_groups        
    while(np_regroup.shape[0]>max_groups) and chi_threshold_flag:

        min_index  =  chisqList.index(min(chisqList))
        merge_index = min_index+1
        np_regroup[min_index,[1,2,3]] = np_regroup[min_index,[1,2,3]]+np_regroup[merge_index,[1,2,3]]
        np_regroup[min_index,0] = np_regroup[merge_index,0]
        np_regroup = np.delete(np_regroup,merge_index,0)
        if np_regroup.shape[0] == 1:
            break

        chisqList = []
        for i in range(0,np_regroup.shape[0]-1):
            a = np_regroup[i,1]
            c = np_regroup[i,2]
            b = np_regroup[i+1,1]
            d = np_regroup[i+1,2]
            chi = (a*d-b*c)**2*(a+b+c+d)/((a+c)*(b+d)*(a+b)*(c+d)+1e-8)
            chi = round(chi,3)
            chisqList.append(chi)  

        chi_threshold_flag = (chisqList < chi_threshold).sum()>0 or np_regroup.shape[0]>max_groups  

    i  =  0
    while (i <= np_regroup.shape[0] - 1):
        n_sample = np_regroup.shape[0]
        if n_sample<2:
            break
        if np_regroup[i, 3]<pct:
            if i == 0:
                np_regroup[i+1,[1,2,3]] = np_regroup[i, [1,2,3]] + np_regroup[i + 1, [1,2,3]]
            elif i == n_sample-1:
                np_regroup[i-1, [1,2,3]] = np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0] = np_regroup[i, 0]
            elif np_regroup[i-1, 3]>= np_regroup[i+1, 3]:
                np_regroup[i+1, [1,2,3]] = np_regroup[i, [1,2,3]] + np_regroup[i+1, [1,2,3]]
            elif np_regroup[i-1, 3]<np_regroup[i+1, 3]:
                np_regroup[i-1, [1,2,3]] = np_regroup[i-1, [1,2,3]] + np_regroup[i, [1,2,3]]
                np_regroup[i-1, 0] = np_regroup[i, 0]
            np_regroup = np.delete(np_regroup,i,0)
            i  =  i - 1
        i  =  i + 1

    cut_point = list(np_regroup[:,0][:-1])
    
    return cut_point
def f_df_num_chisq_group(X,y,
                         p_value = 0.05,
                         pct = 0.05,
                         max_groups = 10,
                         num_least = 10):
    
    
    cut_info = {}
    for col in X.columns:
        group_dict = f_var_num_chisq_group(x = X[col],
                                           y = y,
                                           p_value = p_value,
                                           pct = pct,
                                           max_groups = max_groups,
                                           num_least = num_least)
        cut_info[col] = group_dict
        
    return cut_info
def f_var_num_dtree_group(x,y,
                      criterion = 'gini',
                      min_samples_leaf = 0.05,
                      max_leaf_nodes = 10,
                      min_impurity_decrease = 1e-5):
    
    """
    
    """
    X_var = x
    Y_var = y
    
    X_var.reset_index(drop = True,inplace = True)
    Y_var.reset_index(drop = True,inplace = True)

    Y_var = Y_var[X_var.isnull().apply(lambda x:not x)]
    X_var = X_var[X_var.isnull().apply(lambda x:not x)]
    X_var = np.array(X_var).reshape(-1,1)

    Dtree_model = DTree(criterion = criterion,
              min_samples_leaf = min_samples_leaf,
              min_impurity_decrease = min_impurity_decrease,
              max_leaf_nodes = max_leaf_nodes
              )
 
    Dtree_model.fit(X = X_var,y = Y_var)
    if len(np.argwhere(Dtree_model.tree_.children_left>= 0)) == 0:

        cut_point = []

    else:
        leaf_nodes = np.argwhere(Dtree_model.tree_.children_left<0)
        leaf_nodes = np.squeeze(leaf_nodes)
        tree_paths = {0:[0]}
        for i in range(0,Dtree_model.tree_.node_count):
            if i not in leaf_nodes:

                left_idx = Dtree_model.tree_.children_left[i] 
                right_idx = Dtree_model.tree_.children_right[i]
                tree_paths[left_idx] = tree_paths[i]+list([left_idx])
                tree_paths[right_idx] = tree_paths[i]+list([right_idx])

        leaf_paths = {}    
        for key in tree_paths:
            if key in leaf_nodes:
                leaf_paths[key] = tree_paths[key]

        #tree_paths_dict = {}
        left_value_list = []
        #right_value_list = []
        for item in leaf_paths:

            element = leaf_paths[item]
            element_cnt = len(element)
            left = []
            right = []
            for i in range(0,element_cnt-1):

                threshold = Dtree_model.tree_.threshold[element[i]]
                threshold = round(threshold,6)
                if element[i+1] in Dtree_model.tree_.children_left:
                    left.append(threshold)

                elif element[i+1] in Dtree_model.tree_.children_right:
                    right.append(threshold)

                else:
                    pass

            if len(left) == 0:
                pass

            else:
                left_value = min(left)
                left_value_list.append(left_value)

#             if len(right) == 0:
#                 pass
#             else:
#                 right_value = max(right)
#                 right_value_list.append(right_value)

        cut_point = sorted(left_value_list)  
    
    return cut_point
def f_df_num_dtree_group(X,y,
                         criterion = 'gini',
                         min_samples_leaf = 0.05,
                         max_leaf_nodes = 10,
                         min_impurity_decrease = 1e-5):
    
    cut_info = {}
    for col in X.columns:
        group_dict = f_var_num_dtree_group(x = X[col],
                                           y = y,
                                           criterion = criterion,
                                           min_samples_leaf = min_samples_leaf,
                                           max_leaf_nodes = max_leaf_nodes,
                                           min_impurity_decrease = min_impurity_decrease)
        cut_info[col] = group_dict  
    
    
    return cut_info
def f_var_num_quantile_group(x,q):
    
    if len(x[x.notnull()].unique()) > 0:
        
        cut_points = list(pd.qcut(x,q,retbins = True,duplicates = 'drop')[1]) 
        cut_points = cut_points[1:-1]
        
    else: 
        cut_points = []
    
    return cut_points
def f_df_num_quantile_group(X,q):
    
    cut_info = {}
    
    for col in X.columns:
        cut_info[col] = f_var_num_quantile_group(X[col],q)
        
    return cut_info
def f_var_num_equal_width_group(x,q):
    
    if len(x[x.notnull()].unique()) > 0:
        
        cut_points = list(pd.cut(x,bins = q,retbins = True,duplicates = 'drop')[1])
        cut_points = cut_points[1:-1]
    
    else:
        cut_points = []
    
    return cut_points
def f_df_num_equal_width_group(X,q):
    
    cut_info = {}
    
    for col in X.columns:
        cut_info[col] = f_var_num_equal_width_group(X[col],q)
        
    return cut_info
def f_binning_num(x, cut_point):
    """
    数值分箱
    """
    cut_point_ = [-np.Inf ]+ cut_point + [np.Inf]
    group = pd.cut(x, bins = cut_point_,precision = 10).astype('object')
    group[group.isnull()] = 'null'
    group = group.astype(str)    
    return group
def f_binning_cat(x, cut_point):
    """
    类别分箱
    """
    group = x.map(cut_point)
    group[group.isnull()] = 'null'    
    return group
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
