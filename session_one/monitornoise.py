# coding=gbk

'''
@author: 14941
'''

import pandas as pd

'''
������������ԭ�򣺼������ݷ�����̬�ֲ���ȡֵ��[u-3��,u+3��]�����ڵĸ��ʽ�Ϊ0.03%�����ֿ����Ժ�С����������˼��п������쳣
ԭʼ������������������¼��ֲַ���
    1�����ݻ����ǲ�һ�µģ�
    2�����ݷֲ�Ϊһ��������
    3�����ݷֲ�ȡ����ֵ��
��Ӧ�쳣��
    1��������Ⱥ�㣻
    2�����ֲַ���һ�µĵ㣻
    3������������ֶ��п��ܣ�����ʽ�кܸ���(�д���һ��˼��)
'''

def detectingnoise( data , ratio ):

    '''
    ������������е��쳣����
    :param data: ��������
    :param ratio: ĳһȡֵ��ͬ��������ֵ����������ռ����
    :return:
    '''

    cols_name = data.columns
    cols_name = cols_name[1:]  #��һ��ID�в�Ҫ
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
                # ȱʧֵ���κ�ֵ���ȽϽ��ΪFALSE�����κ�ֵ������������ΪNaNs

                 if(data.ix[row,col]>up_line or data.ix[row,col]<low_line):
                    data.ix[row, col] = 'NA'
                    print(data.ix[row, col])

    return data

# ��������
data = pd.read_excel("train.xlsx")
ratio = 0.7
data1 = detectingnoise(data,ratio)
data1.to_csv("my_train.csv")
