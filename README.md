# data_mining_tool
### 数据挖掘建模工具，对数据处理、数据编码转换、特征筛选、建模开发等常用的方法进行了封装

- src.basic_utils：包含各类功能函数，如计算特征空值率、均值、中位数等，相对目标变量重要性，与目标变量的相关性，iv只，woe编码，判断特征类型等，主要供src.model_var_processor和src.var_filter模块调用
- src.var_bin_utils：包含各类特征分箱方法，如卡方分箱、决策树分箱、分位数分箱和等宽分箱等，供src.model_var_processor调用
- src.var_filter：各类特征筛选方法类模块，根据空值率、单一值、iv大小，和目标变量相关性，随机森林算法确定的特征相对重要性等来过滤筛选变量
- src.var_processor：包含各类特征处理、编码类模块
- src.model_utils：模型处理层面的一些方法类模块，模型选择，超参数搜索选择，模型堆叠功能等
- src.score_utils：评分模型，将概率转换为评分
- src.stat_utils：统计模块，目前只有在逾期信用评分模型领域，特征分组在iv、逾期率、分组占比等方面的统计
- src.utils：其它的一些封装方法，筛选指定类型的特征，模型ks值计算等。
