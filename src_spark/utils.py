# -*- coding: utf-8 -*-


from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F



def compute_features_null_cnt(dataset,inputCols = None):
    ''' 
    就算特征空值数量
    '''
    
    if inputCols is None:
        inputCols = dataset.columns
    
    isnull_lst = []
    for col in inputCols:
        
        isnull_tmp = dataset[col].isNull().astype(IntegerType()).alias(col) 
        isnull_lst.append(isnull_tmp)
    
    accepts_isnull = dataset.select(*isnull_lst)
    
    nullcnt_lst = []
    for col in inputCols:
        
        nullcnt_tmp = F.sum(accepts_isnull[col]).alias(col)
        nullcnt_lst.append(nullcnt_tmp)
        
    
    nullcnt_row = accepts_isnull.select(nullcnt_lst).collect()[0]
    nullcnt_dic = nullcnt_row.asDict()
    
    return nullcnt_dic


def compute_features_null_rate(dataset,inputCols = None):
    
    row_cnt = dataset.count()
    nullcnt_dic = compute_features_null_cnt(dataset,inputCols)
    nullrate_dic = {k:v/row_cnt for k,v in nullcnt_dic.items()}
    
    return nullrate_dic
    


if __name__ == '__main__':
    
    pass


