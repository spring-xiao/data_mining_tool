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
