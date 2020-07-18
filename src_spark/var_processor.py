# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:51:47 2020

@author: xiaoxiang
"""


from pyspark.ml import Transformer,Estimator
from pyspark.sql.types import DoubleType,IntegerType,StringType,FloatType


class DataTypeTransf(Transformer):
    
    """
    
    
    """
    
    def __init__(self,inputColType:dict = None):
        super(DataTypeTransf,self).__init__()
        self.inputColType = dict() if inputColType is None else inputColType
    
    
    def _transform(self,dataset):
        
        colTypeTransf = self._get_cols_type()
        
        cols = dataset.columns
        new_cols = list()
        for col in cols:
            
            if col in colTypeTransf.keys():
                new_cols.append(dataset[col].astype(colTypeTransf[col]))
            
            else:
                new_cols.append(col)
            
        return dataset.select(new_cols)
     
        
    def _get_cols_type(self):
        
        colTypeTransf = dict()
        for k,v in self.inputColType.items():
            
            if v == 'int':
                datatype = IntegerType()
        
            elif v == 'float':
                datatype = FloatType()
                
            elif v == 'str':
                datatype = StringType()
        
            else:
                raise ValueError('特征%s的转换类型要求为int、float、str之一' % k)
                
            colTypeTransf[k] = datatype
          
        return colTypeTransf 


class ImputerByMode(Estimator):
    """
    
    
    """
    def __init__(self,inputCols:list = None):
        super(ImputerByMode,self).__init__()
        self.inputCols = inputCols
        
    
    def _fit(self,dataset):
        
        model_val = {}
        
        for var in self.inputCols:
            
            category_stat = dataset.groupBy(dataset[var]).count().toPandas()
            category_stat_dict = {i[0]:i[1] for i in zip(category_stat[var],category_stat['count'])} 
            mode_val = max(category_stat_dict,key = category_stat_dict.get)
            
            model_val[var] = mode_val
    
        return ImputerByModeModel(model_val)
    
    
class ImputerByModeModel(Transformer):
    """
    
    
    """
    
    def __init__(self,model_val):
        super(ImputerByModeModel,self).__init__()
        self.model_val = model_val
    
    def _transform(self,dataset):
        
        return dataset.fillna(self.model_val)


if __name__ == '__main__':
    
    pass



