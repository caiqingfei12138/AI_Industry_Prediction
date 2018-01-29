Preprocessing.py程序中主要是数据预处理，包括以下内容：
准备数据：删除Ylabel,把train和test数据合并处理；

1，缺失值填充 ，对于类别数据填充众数，对于连续值数据填充均值 ； 
2，利用同一列的方差，把变化不大的列删除；
3，删除特征中取值相同的列 ；
4，按TOOL_ID分离数据 ；
5，把TOOL_ID提取出来，并做one hot编码 ；
6，用正则表达式把时间提取出来，并计算相对于2016 00 00 00 00 00的时间差
7，把5,6的结果合并到每个TOOL_ID数据集中 ；
8，把所有的TOOL_ID合并
9，利用RandomForest（或Xgboost）筛选特征（我没有实现，第一名用xgboos筛选特征，大约选出36个特征，用xgboost预测，结果第一）


predict.ipynb

   0.1*SVR + 0.3*Xgboost(base:svr) +0.3*Xgboost(base:xgb) + 0.3*lgb + 后处理（ 两个分类模型 ）