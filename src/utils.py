from src.basic_utils import get_vars_type


def select_var_by_type(df,uid = None,y = None,var_type = ['int','float']):
    
    if uid is None:
        uid = []
    elif not isinstance(uid,list):
        uid = [uid]
    
    if y is None:
        y = []
    elif not isinstance(y,list):
        y = [y]
    
    if not isinstance(var_type,list):
        var_type = [var_type]
    
    f_types = get_vars_type(df)
    
    if var_type[0] == 'all':
        cols_select = list(f_types.keys())
    else:
        cols_select = [k for k,v in f_types.items() if v in var_type]
    
    cols_select = list(set(cols_select) - set(uid + y))
    
    return cols_select