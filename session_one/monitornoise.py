# coding=gbk

'''
@author: 14941
'''

import pandas as pd

'''
噪声处理，处理原则：假设数据符合正态分布，取值在[u-3σ,u+3σ]区间内的概率仅为0.03%，出现可能性很小如果，出现了极有可能是异常
原始数据特征大体呈现以下几种分布：
    1，数据基本是不一致的；
    2，数据分布为一个常数；
    3，数据分布取有限值；
对应异常：
    1，出现离群点；
    2，出现分布不一致的点；
    3，这个以上两种都有可能，处理方式有很复杂(有待进一步思考)
'''

def detectingnoise( data , ratio ):

    '''
    检测特征矩阵中的异常数据
    :param data: 特征矩阵
    :param ratio: 某一取值不同特征数据值在样本中所占比例
    :return:
    '''

    cols_name = data.columns
    cols_name = cols_name[1:]  #第一行ID列不要
    data = data[cols_name]
    ndim = data.shape
    ratio = ratio*ndim[0]
    rows = ndim[0]
    for col in range(len(cols_name)):
        col_unique = data[cols_name[col]].unique()
        col_unique_len=len(col_unique)
        if(col_unique_len>ratio):
            mean = data[cols_name[col]].mean(axis=0)
            std  = data[cols_name[col]].std(axis=0)
            up_line = mean + 3*std
            low_line = mean -3*std
            for row in range(rows):
                # 缺失值和任何值做比较结果为FALSE，和任何值做算术运算结果为NaNs

                 if(data.ix[row,col]>up_line or data.ix[row,col]<low_line):
                    data.ix[row, col] = 'NA'
                    print(data.ix[row, col])

    return data

# 加载数据
data = pd.read_excel("train.xlsx")
ratio = 0.7
data1 = detectingnoise(data,ratio)
data1.to_csv("my_train.csv")
