# -*- coding: utf-8 -*-

import pandas as pd
from sqlalchemy import create_engine
from impala.dbapi import connect
from impala.util import as_pandas
from pymongo import MongoClient
import re
from decimal import Decimal


class mySqlConnect():
    '''
    连接MySQL的类
    例如：
    con = MySQLConnect()
    r = con.read("select 123")
    
    
    方法：
    read(statement)：读取数据
    write(statement)：写入数据
    close()：关闭所有连接
    '''
    def __init__(self,user,password,database,host = 'localhost',port = '3306'):
        
        self.user = user
        self.password = password
        self.database = database
        self.host = host
        self.port = port 
        self._driver = create_engine('mysql+pymysql://{user}:{password}@{host}:{port}/{database}'.\
                            format(user = self.user,
                                   password = self.password,
                                   host = self.host,
                                   port = self.port, 
                                   database = self.database, 
                                   encoding = 'utf-8')
                            )

    def close(self):
        self._driver.dispose()

    def connect_test(self):
        result = self.read("select 123")
        #return result
        return len(result) == 1
    
    def read(self, statement): 
        with self._driver.connect() as con:
            rs = con.execute(statement)
            result = rs.fetchall()
            return result
        
    def execute(self, statement):
        with self._driver.connect() as con:
            rs = con.execute(statement)        
    
    def write(self, table_name, dic, debug = 0):
        column_expr =  ', '.join(['`' + k + '`' for k in dic.keys()])
        # value_expr =  ', '.join(['\'' + str(k) + '\'' for k in dic.values()])
        value_expr =  ', '.join(['%s' for k in dic.values()])
        insert_sql = '''insert into `{table_name}` ({column_expr})
            values({value_expr})'''.format(table_name = table_name, column_expr = column_expr, value_expr = value_expr)
        if debug > 0:
            print(insert_sql)
        # result = self._driver.execute(insert_sql)
        result = self._driver.execute(insert_sql, tuple(dic.values()))
        return result.rowcount
    
    def read_df(self, statement):
        result = pd.read_sql_query(statement, self._driver)
        return result
    
    def write_df(self, table_name, df,if_exists = 'append'):
        df.to_sql(table_name, self._driver, if_exists = if_exists, index = False)


class hiveConnect():
    
    def __init__(self,
                 user = None,
                 password = None,
                 host = '10.101.40.229',
                 port = 21050,
                 database = 'db_broker_label',
                 auth_mechanism = 'NOSASL'):
        
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.auth_mechanism = auth_mechanism
        self.create_connect()


    def create_connect(self):
        
        self.conn = connect(self.host,self.port,self.database,self.user,self.password,self.auth_mechanism)
        self.cursor = self.conn.cursor()
        
    def close_connect(self):
        
        self.cursor.close()
        self.conn.close()

    def read(self,statement):
        
        self.create_connect()
        self.cursor.execute(statement)
        dat = self.cursor.fetchall()
        self.close_connect()
        return dat
    
    def read_df(self,statement):
        
        self.create_connect()
        self.cursor.execute(statement)
        df = as_pandas(self.cursor)
        self.close_connect()
        
        return df
    
    def write_df(self,table_name,df,cols_type = dict(),once_cnt = 1000):
        
        self.create_connect()    
        if not cols_type:
            cols_type = get_cols_type(df)
        
        data_cnt = df.shape[0]
        iters = int(data_cnt/once_cnt) if data_cnt%once_cnt == 0 else int(data_cnt/once_cnt) + 1
        
        sql_insert = "insert into {} values({})"
        
        for i in range(iters):

            start = i*once_cnt
            end = data_cnt if i == iters-1 else (i + 1)*once_cnt
            df_sub = df.iloc[start:end]
            dat_lst = ["(" + self._concat_cols_to_string(i,cols_type) + ")" for i in df_sub.to_dict("records")]
            sql_insert = sql_insert.format(table_name,",".join(dat_lst))
            self.cursor.execute(sql_insert)
        
        self.close_connect()
                    
    def delete(self,table_name):
        self.create_connect()
        sql_delete = "truncate table {}".format(table_name)
        self.cursor.execute(sql_delete)
        self.close_connect()
        
    def _get_val_by_type(self,col_name,col_value,col_type_dic):
        
        if pd.isnull(col_value):
            res = 'null'
        
        elif col_type_dic[col_name] in ['string','str'] and  len(re.findall('^[0-9.]+$',str(col_value))) > 0:
            res = Decimal(str(col_value)).normalize()
            res = "'" + str(res) + "'"
        elif col_type_dic[col_name] in ['string','str']:
            res = "'" + str(col_value) + "'"
        elif col_type_dic[col_name] in ['bigint','int']:
            res = int(col_value)
        elif col_type_dic[col_name] in ['float']:
            res = float(col_value)
        else:
            raise ValueError("{}的类型未找到".format(col_name))
            
        return res
    
    def _concat_cols_to_string(self,val_dic,col_type_dic):
        
        val_str = ''
        for key,val in val_dic.items():
            res = self._get_val_by_type(key,val,col_type_dic)
            val_str += str(res) + ','
    
        val_str = val_str[0:-1]
        return val_str


class mongodbConnect():
    
    def __init__(self,host,port,database,user = None,password = None):
        self.user = user
        self.password = password
        self.host = host
        self.port = port 
        self.database = database
        #self.collection = collection
        self.conn_client = MongoClient(host = self.host,port = self.port)
        
        if self.database in self.conn_client.list_database_names():
            self.conn_db = self.conn_client[self.database]
        
        else:
            raise ValueError("数据库名{}不存在".format(self.database))
        
        #self.conn_collection = self.conn_db[self.collection]
    
    
    def insert_one(self,data,collection):
        
        if collection in self.conn_db.list_collection_names():
            
            conn_collection = self.conn_db[collection]
            conn_collection.insert_one(data)
            
        else:
            raise ValueError("集合{}不存在".format(collection))
            
        
    def insert_many(self,data_lst,collection,once_cnt = 1000):
        
        if collection in self.conn_db.list_collection_names():
            conn_collection = self.conn_db[collection]
            data_len = len(data_lst)
            iters = int(data_len/once_cnt) if data_len%once_cnt == 0 else int(data_len/once_cnt) + 1
            for i in range(iters):
                start = i*once_cnt
                end = data_len if i == iters-1 else (i+1)*once_cnt
                conn_collection.insert_many(data_lst[start:end])
            
        else:
            raise ValueError("集合{}不存在".format(collection))
        
    
    def read(self,statement,collection):
        
        conn_collection = self.conn_db[collection]
        data = conn_collection.find(statement)
        data = list(data)
        return data
    
    def read_df(self,statement,collection):
       
        conn_collection = self.conn_db[collection]
        data = conn_collection.find(statement)
        data = list(data)
        df = pd.DataFrame(data)
        return df
    
    def create_collection(self,name):
        
        if name not in self.conn_db.list_collection_names():
            self.conn_db.create_collection(name)
        
        else:
            print("集合{}已存在".format(name))
            
    def drop_collection(self,name):
        
        if name in self.conn_db.list_collection_names():
            self.conn_db.drop_collection(name)
            
        else:
            print("要删除的集合{}不存在".format(name))
   
    
def get_cols_type(df):
    
    cols_type = dict(df.dtypes.apply(lambda x:str(x)))
    f_types = {}
    for k,v in cols_type.items():
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
    

#------------------------------------------------------------------------------
  
if __name__ == '__main__':
    pass
    
    






