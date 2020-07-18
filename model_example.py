# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:47:03 2020

@author: xiaoxiang
"""

#***********************************************
#*** 加载包

import pandas as pd
# import pickle
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.basic_utils import get_vars_type
from src.utils import select_var_by_type
from sklearn.metrics import roc_auc_score
from src.utils import get_ks
from src.stat_utils import get_df_iv_info

from src.var_processor import (
        FillNullVarNum,
        FillNullVarStr,
        OneHotEnc,
        BinVarNum
        )

from src.score_utils import ProbaToScore

#********************************************
#****导入数据
data = pd.read_csv('data/accepts.csv')

'''
'application_id' : 申请者ID
'account_number' : 帐户号
'bad_ind' : 是否违约
'vehicle_year' : 汽车购买时间
'vehicle_make' : 汽车制造商
'bankruptcy_ind' : 曾经破产标识
'tot_derog' : 五年内信用不良事件数量(比如手机欠费消号)
'tot_tr' : 全部帐户数量
'age_oldest_tr' : 最久账号存续时间(月)
'tot_open_tr' : 在使用帐户数量
'tot_rev_tr': 在使用可循环贷款帐户数量(比如信用卡)
'tot_rev_debt' : 在使用可循环贷款帐户余额(比如信用卡欠款)
'tot_rev_line': 可循环贷款帐户限额(信用卡授权额度)
'rev_util' : 可循环贷款帐户使用比例(余额/限额)
'fico_score' : FICO打分
'purch_price' : 汽车购买金额(元)
'msrp' : 建议售价
'down_pyt' : 分期付款的首次交款
'loan_term' : 贷款期限(月)
'loan_amt': 贷款金额
'ltv' : 贷款金额/建议售价*100
'tot_income' : 月均收入(元)
'veh_mileage': 行使历程(Mile)
'used_ind' : 是否二手车
'''

cols = data.columns.tolist()
except_var = ['account_number']

uid = 'application_id'
y_var = 'bad_ind'
x_var = set(cols) - set([uid] + [y_var]+ except_var)
x_var = list(x_var)

x_var_type = get_vars_type(data[x_var])  #获取各字段类型
x_var_num = select_var_by_type(data[x_var],var_type = ['int','float'])
x_var_str = select_var_by_type(data[x_var],var_type = ['str'])

#*************************************************
#***数据划分
X_train,X_test,y_train,y_test = train_test_split(
        data[x_var],data[y_var],test_size = 0.3,random_state = 42)


X_train.reset_index(drop = True,inplace = True)
X_test.reset_index(drop = True,inplace = True)
y_train.reset_index(drop = True,inplace = True)
y_test.reset_index(drop = True,inplace = True)


#**************************************************
#***数据处理，pipeline管道流处理

fill_null_var_num = FillNullVarNum(cols_lst = x_var_num,uid = uid,y = y_var,fill_thres = 0.05)
fill_null_var_str = FillNullVarStr(cols_lst = x_var_str,uid = uid,y = y_var,fill_thres = 0.05)
one_hot_enc = OneHotEnc(uid = uid,y = y_var)

steps = [('fill_null_var_num',fill_null_var_num),
         ('fill_null_var_str',fill_null_var_str),
         ('one_hot_enc',one_hot_enc)]

var_pro_pip= Pipeline(steps)
X_train_model = var_pro_pip.fit_transform(X = X_train,y = y_train)
X_test_model = var_pro_pip.transform(X_test)

#**************************************************
#***模型建立

params = {'max_depth' : 2,
          'leraning_rate' : 0.0001,
          'n_estimators': 50,
          'subsample' : 0.9,
          'colsample_bytree' : 0.9
          }

xgb_model = XGBClassifier(**params)
xgb_model.fit(X = X_train_model,y = y_train)

y_pred_train = xgb_model.predict_proba(data = X_train_model)[:,1]
y_pred_test =  xgb_model.predict_proba(data = X_test_model)[:,1]

auc_train = roc_auc_score(y_true = y_train,y_score = y_pred_train)
ks_train = get_ks(y_true = y_train,y_score = y_pred_train)

auc_test = roc_auc_score(y_true = y_test,y_score = y_pred_test)
ks_test = get_ks(y_true = y_test,y_score = y_pred_test)

print('auc_train',round(auc_train,3))
print('ks_train',round(ks_train,3))
print('auc_test',round(auc_test,3))
print('ks_test',round(ks_test,3))


#**************************************************
#***评分制定

score_pro = ProbaToScore(transf_reverse = False)
score_pro.fit()
score_train = score_pro.transform(X = y_pred_train)
score_test = score_pro.transform(X = y_pred_test)

score_train_df = pd.DataFrame(score_train,columns = ['score'])
score_test_df = pd.DataFrame(score_test,columns = ['score'])

score_bin = BinVarNum(cols_lst = ['score'],num_least = 10,bin_type = 'chisq')
score_train_bin= score_bin.fit_transform(X = score_train_df,y = y_train)
score_test_bin= score_bin.transform(X = score_test_df)

score_train_info = get_df_iv_info(X = score_train_bin,y = y_train)
score_test_info = get_df_iv_info(X = score_test_bin,y = y_test)



#score_train_info.to_csv('result\score_train_info.csv',index = False)
#score_train_info.to_csv('result\score_test_info.csv',index = False)






