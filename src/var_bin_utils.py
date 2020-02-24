# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:47:19 2019

@author: xiaoxiang
"""


import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.tree import  DecisionTreeClassifier as DTree


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


