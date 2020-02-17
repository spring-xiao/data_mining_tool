# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2019

@author: xiaox
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone

class ModelSelect(BaseEstimator):
    
    def __init__(self,model_lst:dict = {},scoring = 'neg_mean_squared_error',cv = 5):
        self.model_lst = model_lst
        self.scoring = scoring
        self.cv = cv
        self.model_score = {}
        self.fit_status = False
        
    def fit(self,X,y = None):
        for k,model in self.model_lst.items():
            if self.scoring == 'neg_mean_squared_error':
                model_score = cross_val_score(model,X = X,y = y,scoring = self.scoring, cv = self.cv)
                self.model_score[k] = np.sqrt(-model_score)
        
        self.fit_status = True    
            
    def transform(self,select_thres = 0.125):
        
        self.metrics = {k:[round(v.mean(),6),round(v.std(),6)] for k,v in self.model_score.items() if v.mean() <= select_thres}
        return self.metrics
    
    def fit_transform(self,X,y,select_thres = 0.125):
        
        self.fit(X,y)
        self.metrics = self.transform(select_thres = select_thres)
        return self.metrics
    
class ParamsSelect(BaseEstimator):
    
    def __init__(self,model,scoring = 'neg_mean_squared_error',cv = 5):
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.fit_status = False
    
    def fit(self,X,y,params_grid,n_jobs = 3):

        grid_search = GridSearchCV(self.model,params_grid,cv = self.cv,scoring = self.scoring,n_jobs = n_jobs)
        grid_search.fit(X,y)
        self.best_estimator = grid_search.best_estimator_ 
        self.best_params = grid_search.best_params_ 
        self.cv_results_ = grid_search.cv_results_
        if self.scoring == 'neg_mean_squared_error':
            self.best_score = np.sqrt(-grid_search.best_score_)
            
        else:
            self.best_score = grid_search.best_score_
            
        self.fit_status = True
        
    def transform(self):
        
        return self.best_params
    
    def fit_transform(self,X,y,params_grid,n_jobs = 3):
        
        self.fit(X,y,params_grid,n_jobs)
        best_params = self.transform()
        return best_params
    
    
class AvgWeightRegressor(BaseEstimator,TransformerMixin):
    
    def __init__(self,weights:list = [],fit_type  = 'default'):
        self.weights = weights
        self.fit_type = fit_type
        self.fit_status = False
    
    def fit(self,X,y = None):
        if self.fit_type == 'default':
            self.weights = [1/X.shape[1]]*X.shape[1]
        
        elif self.fit_type == 'weight':
            
            X = np.array(X)
            root_mean_error = []
            for i in range(X.shape[1]):
                error = np.sqrt(mean_squared_error(X[:,i] ,y))
                root_mean_error.append(error)
            self.weights = root_mean_error/np.sum(root_mean_error)
        else:
            pass
            
    def predict(self,X):
        
        X = np.array(X)
        pred_val = np.dot(X,self.weights)
        return pred_val    

class StackModel(BaseEstimator,TransformerMixin):
    
    def __init__(self,model_lst:list = [],meta_model = None,cv = 5):
        self.model_lst = model_lst
        self.meta_model = meta_model
        self.cv = cv
        self.fit_status = False
    
    def fit(self,X,y):  
        
        kf = KFold(n_splits = self.cv, shuffle = True, random_state = 42)
        self.renew_model_lst = [list() for i in self.model_lst]
        pred_value = np.zeros((X.shape[0],len(self.model_lst)))
        
        for i,model in enumerate(self.model_lst):
            for train_idx,val_idx in kf.split(X = X,y = y):
                model_tmp = clone(model)
                model_tmp.fit(X = X.iloc[train_idx],y = y.iloc[train_idx])
                self.renew_model_lst[i].append(model_tmp)
                pred_value[val_idx,i] = model_tmp.predict(X.iloc[val_idx]) 
                
        self.meta_model.fit(X = pred_value,y = y)
        self.fit_status = False
        
    
    def predict(self,X):
        
        mid_pred_value = []
        for models in self.renew_model_lst:
            mid_pred_value.append(np.column_stack([model.predict(X) for model in models]).mean(axis = 1))
        
        mid_pred_value = np.column_stack(mid_pred_value)
        pred_value = self.meta_model.predict( X = mid_pred_value)
        
        return pred_value
    
    