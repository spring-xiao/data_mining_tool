# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2020

@author: xiaox
"""


from sklearn.base import BaseEstimator,TransformerMixin
from math import log
import pandas as pd

class ProbaToScore(BaseEstimator,TransformerMixin):
    
    def __init__(self,odds_val = 1/10,score_val = 500, pdo = 50,
                 score_scaled_min = 300,score_scaled_max = 850,score_scaled = True):
        self.odds_val = odds_val
        self.score_val = score_val
        self.pdo = pdo
        self.score_scaled_min = score_scaled_min
        self.score_scaled_max = score_scaled_max
        self.score_scaled = score_scaled
        self.fit_status = False
        
    def fit(self):
        self.B = self.pdo/log(2)
        self.A = self.score_val + self.B*log(self.odds_val)
        self.fit_status = True
    
    def transform(self,X):
        
        X_odds = pd.Series(X).apply(lambda x: 0.9999/(1-0.9999) if x >=1 else x/(1-x) )
        score = X_odds.apply(lambda x:round(self.A - self.B*log(x)))
        
        self.sample_score_min = min(score)
        self.sample_score_max = max(score)
        
        if self.score_scaled:
            score = score.apply(lambda x:self.__scaled_score__(x))
            
        return score
    
    def fit_transform(self,X):
        
        self.fit()
        score = self.transform(X)
        return score
    
    def get_score(self,val):
        
        val_odds = 0.9999/(1-0.9999) if val >= 1 else val/(1-val)
        score = round(self.A - self.B*log(val_odds))
        if self.score_scaled:
            score = self.__scaled_score__(score)

        return score
    
    def __scaled_score__(self,score):
        
        k = (self.score_scaled_max - self.score_scaled_min)/(self.sample_score_max - self.sample_score_min)
        b = self.score_scaled_max - k*self.sample_score_max
        score = k*score + b
        score = round(score)
        if score > self.score_scaled_max:
            score = self.score_scaled_max
        elif score < self.score_scaled_min:
            score = self.score_scaled_min
        
        return score


