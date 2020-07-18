# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:20:19 2020

@author: xiaoxiang
"""

from sklearn.metrics import roc_auc_score,roc_curve

def get_ks(y_true,y_score):
    
    fpr, tpr, thresholds = roc_curve(y_true = y_true, y_score = y_score)
    ks = max(abs(tpr - fpr))
    return ks
