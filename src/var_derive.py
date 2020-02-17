# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:11:34 2019

@author: xiaox
"""


from sklearn.base import BaseEstimator,TransformerMixin
from src.utils import select_var_by_type
import copy

class FeatureDerive(BaseEstimator,TransformerMixin):
    
    require_var_type = ['all']
    
    def __init__(self,cols_lst:list = [],uid = None,y = None,derive_type = None):
        self.cols_lst = cols_lst
        self.uid = uid
        self.y = y
        self.derive_type =derive_type
        self.fit_status = False
    
    def fit(self,X,y = None): 
        if len(self.cols_lst) == 0:
            self.cols_lst = select_var_by_type(X,uid = self.uid,y = self.y,var_type = self.require_var_type)
        else:
            self.cols_lst = [col for col in self.cols_lst if col in X.columns.tolist()]  
        self.fit_status = True
        
    def transform(self,X):
        df = copy.deepcopy(X)
        
        df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]   
        df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"]

        df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * df["OverallQual"]
        df["+_GrLivArea_OverallQual"] = df["GrLivArea"] * df["OverallQual"]
#         df["+_oMSZoning_TotalHouse"] = df["oMSZoning"] * df["TotalHouse"]
#         df["+_oMSZoning_OverallQual"] = df["oMSZoning"] + df["OverallQual"]
#         df["+_oMSZoning_YearBuilt"] = df["oMSZoning"] + df["YearBuilt"]
#         df["+_oNeighborhood_TotalHouse"] = df["oNeighborhood"] * df["TotalHouse"]
#         df["+_oNeighborhood_OverallQual"] = df["oNeighborhood"] + df["OverallQual"]
#         df["+_oNeighborhood_YearBuilt"] = df["oNeighborhood"] + df["YearBuilt"]
        df["+_BsmtFinSF1_OverallQual"] = df["BsmtFinSF1"] * df["OverallQual"]

#         df["-_oFunctional_TotalHouse"] = df["oFunctional"] * df["TotalHouse"]
#         df["-_oFunctional_OverallQual"] = df["oFunctional"] + df["OverallQual"]
        df["-_LotArea_OverallQual"] = df["LotArea"] * df["OverallQual"]
        df["-_TotalHouse_LotArea"] = df["TotalHouse"] + df["LotArea"]
#         df["-_oCondition1_TotalHouse"] = df["oCondition1"] * df["TotalHouse"]
#         df["-_oCondition1_OverallQual"] = df["oCondition1"] + df["OverallQual"]


        df["Bsmt"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"]
        df["Rooms"] = df["FullBath"]+df["TotRmsAbvGrd"]
        df["PorchArea"] = df["OpenPorchSF"]+df["EnclosedPorch"]+df["3SsnPorch"]+df["ScreenPorch"]
        df["TotalPlace"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"] +     df["OpenPorchSF"]+df["EnclosedPorch"]+df["3SsnPorch"]+df["ScreenPorch"]
        
        return df
    def fit_transform(self,X,y = None):
        
        self.fit(X)
        df = self.transform(X)
        return df
    


