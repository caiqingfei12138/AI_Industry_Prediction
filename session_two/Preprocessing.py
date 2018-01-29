# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:05:41 2018

@author: 14941
"""

'''
工业数据预处理一条龙运行
'''

# load library
import warnings
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

warnings.filterwarnings('ignore')


# 参考： http://www.itkeyword.com/doc/5831929987635309082/impute-categorical-missing-values-in-scikit-learn
'''
普适性的方法：
对于缺失的Object填充众数；
对于缺失的non-Object填充众数
'''
def delmorefeature( tool ):
    missing_count = tool.apply(lambda x: sum(x.isnull()))
    # 1，删除缺失较多的特征
    more_missing = missing_count[missing_count>=1511]
    more_missing_index = list(more_missing.index)
    tool.drop(more_missing_index,axis=1,inplace = True)
    return tool
    
class SeriesImputer_one(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.  

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'): self.fill = X.value_counts().index[0]
        else                            : self.fill = X.value_counts().index[0]
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

'''
普适性的方法：
对于缺失的Object填充均值；
对于缺失的non-Object填充众数
'''



class SeriesImputer_two(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.  

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'): self.fill = X.value_counts().index[0]
        else                            : self.fill = X.mean()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# --------------------------------------------------------TOOL处理：
def fill_missing (tool):
    '''
    根据不同的场景填充，
    1，如果特征是类别，等级等特征，众数填充；
    2，如果特征是连续数据值，观察数据分布，均值填充，这些只要同过Unique来体现。
    '''
    cols = tool.columns
    for col in cols:
        len_unique = len(tool[col].unique())
        if ( len_unique<20): 
            imp=SeriesImputer_one()
            imp.fit(tool[col])
            s = imp.transform(tool[col]) 
            tool[col] = s
        else:
            imp=SeriesImputer_two( )
            imp.fit(tool[col])
            s = imp.transform(tool[col]) 
            tool[col] = s
    return tool


# 不用库自带，库自带有缺陷，处理之后字段名，变化了，不利于后续的分析
# 这里写一个普适性的方法分析方法：
class SeriesVariance_analyse(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, 不进行方差计算
        If the Series is not of dtype Object, 进行方差计算

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'): self.var = None
        else                            : self.var = X.var()
        return self

    def transform(self, X, y=None):
        if X.dtype == np.dtype('O'): 
            return None
        return X.var( )


# 调用上面写的针对series的方差分析函数
def del_one_cols( data,thresholds):
    '''
    通过方差分析，之前用过 
    '''
    cols = data.columns
    for col in cols :
        s = data[col]
        oj  = SeriesVariance_analyse( )   # Initialize the imputer
        oj.fit(s)              # Fit the imputer
        var = oj.transform(s)   # Get a new series
        if (var==None):
            data[col] = s
        elif (var<=thresholds):
            data.drop( col,axis=1,inplace=True )  
            
    return data  

'''
删除不同特征中的相同值
'''
def del_muti_cols( tool1_1 ):
    '''
    对比两个特征列的平均欧氏距离，小于threshold的列，两列中任意删除一列 ,编程技巧：用递归的方式去解决，这样编程会更方便
    '''
    cols = tool1_1.columns
    tool1_2 = tool1_1.T
    flag = tool1_2.duplicated( )
    drop_cols = cols[flag]
    tool1_2=tool1_2.T
    tool1_2.drop( drop_cols,axis=1,inplace=True )
    return tool1_2

import re



def as_num(x):
    y='{:.0f}'.format(x) # 0f表示保留5位小数点的float型
    return (y)


def extract_time ( data ) :
    '''
    用正则表达式把每道工序中的数据提取出来
    '''
    raw_data = data.ix[:,2:]
    feature_cols = raw_data.columns
    date_cols = [ ] # store date data
    rule =  r'^2017|2016'
    compiled_rule = re.compile(rule)   
    for feture_index in range(len(feature_cols)):
        if( compiled_rule.findall(as_num(raw_data.ix[0,feture_index]))==['2017'] or 
            compiled_rule.findall(as_num(raw_data.ix[0,feture_index]))==['2016']):
            if( len(as_num(raw_data.ix[0,feture_index]))==8  or 
                len(as_num(raw_data.ix[0,feture_index]))==14 or
                len(as_num(raw_data.ix[0,feture_index]))==16 ):
            
                date_cols.append(feature_cols[feture_index]) 
    
    date = data[date_cols]
    data.drop(date_cols,axis=1,inplace = True)
            
    return data ,date


# 对分离时间后的数据data做oneHot编码转化：
#def oneHot_encode( tool ) :
#    '''
#    编码针对的列不限于与字符，具有等级，类别的列，也用one_Hot编码
#    '''
#    cols = tool.columns
#    oneHot_cols = [ ]
#    for col in cols:
#        len_unique = len(tool[col].unique())
#        if ( len_unique<=10): 
#            oneHot_cols.append(col)
#    
#   
#    one_hot = pd.get_dummies(tool, columns=oneHot_cols)
#    one_hot = one_hot.reset_index(drop=True)
#    return one_hot



# 对分离时间后的数据data做oneHot编码转化：
# from sklearn.preprocessing import OneHotEncoder
def oneHot_encode( tool ) :
    '''
    编码针对的列不限于与字符，具有等级，类别的列，也用one_Hot编码
    '''
    cols = tool.columns
    cols = cols[:2]
    oneHot_cols = [ ]
    for col in cols:
        len_unique = len(tool[col].unique())
        if ( len_unique<=10): 
            oneHot_cols.append(col)
    
   
    one_hot = pd.get_dummies(tool, columns=oneHot_cols)
    one_hot = one_hot.reset_index(drop=True)
    return one_hot


def moreoneHot_encode( tool ) :
    '''
    编码针对的列不限于与字符，具有等级，类别的列，也用one_Hot编码
    '''
    cols = tool.columns
    oneHot_cols = [ ]
    for col in cols:
        len_unique = len(tool[col].unique())
        if ( len_unique<=10): 
            oneHot_cols.append(col)
    
   
    one_hot = pd.get_dummies(tool, columns=oneHot_cols)
    one_hot = one_hot.reset_index(drop=True)
    return one_hot

#时间中存在异常，很多都是2017的时间突然冒出一个2016，处理方式字符截取 例如：20170616 --- 截取 前五位后：为：616   ‘20170711004950’截取为：711004950
# 有的工序中没有时间特征，因此在处理前需要做DataFrame判空

# 基准时间为2016 00 00 00 00 00 ，最小时间尺度为秒（s）
def deal_time( date ,tool ) :
    '''
    对时间做一定的变换。
    '''
    if(date.empty == False):
        size = date.shape
        for col in range(size[1]):
            for row in range(size[0]):
                s = as_num(date.ix[row,col])
                if(len(s)==8):                                           # 将8位数的时间转化为秒
                    if(s[0:4]=='2016'):
                        time = 316*24*3600 + (int(s[4:6])*30 + int(s[6:8]))*3600*24
                    else:
                        time = (int(s[4:6])*30 + int(s[6:8]))*3600
                else:
                    if(s[0:4]=='2016'):
                        time = 316*24*3600 + (int(s[4:6])*30 + int(s[6:8]))*3600*24 + (int(s[8:10])*60+ int(s[10:12]))*60 + int(s[12:14])
                    else:
                        time = (int(s[4:6])*30 + int(s[6:8]))*3600*24 + (int(s[8:10])*60+ int(s[10:12]))*60 + int(s[12:14])

                date.ix[row,col] = time
        diff_date = (date.max(axis=1)) - (date.min(axis=1))
        tool['diffdate'] = diff_date
        
    return tool



import time
start = time.clock()
for index in range(12):
    filepath = 'H:/java/python/src/machinelearning/seconddata/split_data/TOOL'+str(index+1)+'.xlsx'
    
    tool  = pd.read_excel(filepath)
    
    tool = delmorefeature( tool )
    
    tool1 = fill_missing(tool)
    tool1.head(5)
    
    #   测试
    tool1_1 = del_one_cols(tool1 ,0.005 )
    tool1_1.shape
    
    # 测试通过
    tool1_2=del_muti_cols(tool1_1)
    tool1_2.shape
    
    # 测试成功
    # 测试成功
    print(tool1_2.shape)
    data1 ,date1 = extract_time( tool1)
    print(data1.head(5))
    print(date1.head(5)) 
    
    # 测试成功
    oneHot=oneHot_encode(data1)
    # oneHot.to_csv('H:/java/python/src/machinelearning/seconddata/preprocessing/TOOL1_oneHot.csv',index=None)
    oneHot.head(5)
    
    # 测试
    new_tool = deal_time(date1,oneHot)
    oneHot.to_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL'+str(index+1)+'.csv',index=None)

TOOL1 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL1.csv')
print(TOOL1.shape)
TOOL2 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL2.csv')
TOOL2.drop('ID',axis=1,inplace=True)
print(TOOL2.shape)
TOOL3 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL3.csv')
TOOL3.drop('ID',axis=1,inplace=True)
print(TOOL3.shape)
TOOL4 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL4.csv')
TOOL4.drop('ID',axis=1,inplace=True)
print(TOOL4.shape)
TOOL5 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL5.csv')
TOOL5.drop('ID',axis=1,inplace=True)
print(TOOL5.shape)
TOOL6 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL6.csv')
TOOL6.drop('ID',axis=1,inplace=True)
print(TOOL6.shape)
TOOL7 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL7.csv')
TOOL7.drop('ID',axis=1,inplace=True)
print(TOOL7.shape)
TOOL8 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL8.csv')
TOOL8.drop('ID',axis=1,inplace=True)
print(TOOL8.shape)
TOOL9 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL9.csv')
TOOL9.drop('ID',axis=1,inplace=True)
print(TOOL9.shape)
TOOL10 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL10.csv')
TOOL10.drop('ID',axis=1,inplace=True)
print(TOOL10.shape)
TOOL11 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL11.csv')
TOOL11.drop('ID',axis=1,inplace=True)
print(TOOL11.shape)
TOOL12 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL12.csv')
TOOL12.drop('ID',axis=1,inplace=True)
print(TOOL12.shape)
# write .csv
data = pd.concat([TOOL1,TOOL2,TOOL3,TOOL4,TOOL5,TOOL6,TOOL7,TOOL8,TOOL9,TOOL10,TOOL11,TOOL12], axis=1, join_axes=[TOOL1.index])
# data = pd.concat([TOOL1,TOOL2,TOOL3], axis=1, join_axes=[TOOL1.index])
data.to_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/data.csv')



for index in range(12):
    filepath = 'H:/java/python/src/machinelearning/seconddata/split_data/TOOL'+str(index+1)+'.xlsx'
    
    tool  = pd.read_excel(filepath)
    
    tool = delmorefeature( tool )
    
    tool1 = fill_missing(tool)
    tool1.head(5)
    
    #   测试
    tool1_1 = del_one_cols(tool1 ,0.0005 )
    tool1_1.shape
    
    # 测试通过
    tool1_2=del_muti_cols(tool1_1)
    tool1_2.shape
    
    # 测试成功
    # 测试成功
    print(tool1_2.shape)
    data1 ,date1 = extract_time( tool1)
    print(data1.head(5))
    print(date1.head(5)) 
    
    # 测试成功
    oneHot=moreoneHot_encode(data1)
    # oneHot.to_csv('H:/java/python/src/machinelearning/seconddata/preprocessing/TOOL1_oneHot.csv',index=None)
    oneHot.head(5)
    
    # 测试
    new_tool = deal_time(date1,oneHot)
    oneHot.to_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id'+str(index+1)+'.csv',index=None)


TOOL1 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id1.csv')
print(TOOL1.shape)
TOOL2 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id2.csv')
TOOL2.drop('ID',axis=1,inplace=True)
print(TOOL2.shape)
TOOL3 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id3.csv')
TOOL3.drop('ID',axis=1,inplace=True)
print(TOOL3.shape)
TOOL4 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id4.csv')
TOOL4.drop('ID',axis=1,inplace=True)
print(TOOL4.shape)
TOOL5 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id5.csv')
TOOL5.drop('ID',axis=1,inplace=True)
print(TOOL5.shape)
TOOL6 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id6.csv')
TOOL6.drop('ID',axis=1,inplace=True)
print(TOOL6.shape)
TOOL7 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id7.csv')
TOOL7.drop('ID',axis=1,inplace=True)
print(TOOL7.shape)
TOOL8 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id8.csv')
TOOL8.drop('ID',axis=1,inplace=True)
print(TOOL8.shape)
TOOL9 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id9.csv')
TOOL9.drop('ID',axis=1,inplace=True)
print(TOOL9.shape)
TOOL10 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id10.csv')
TOOL10.drop('ID',axis=1,inplace=True)
print(TOOL10.shape)
TOOL11 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id11.csv')
TOOL11.drop('ID',axis=1,inplace=True)
print(TOOL11.shape)
TOOL12 = pd.read_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/TOOL_id12.csv')
TOOL12.drop('ID',axis=1,inplace=True)
print(TOOL12.shape)
# write .csv
data = pd.concat([TOOL1,TOOL2,TOOL3,TOOL4,TOOL5,TOOL6,TOOL7,TOOL8,TOOL9,TOOL10,TOOL11,TOOL12], axis=1, join_axes=[TOOL1.index])
# data = pd.concat([TOOL1,TOOL2,TOOL3], axis=1, join_axes=[TOOL1.index])
data.to_csv('H:/java/python/src/machinelearning/seconddata/afterPreprocessing/data_id.csv')


end = time.clock()
print ("runtime: %f s" % (end - start))