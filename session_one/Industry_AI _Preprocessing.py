# coding=gbk
'''
Created on 2017��12��23��

@author: 14941
'''
import re
import pandas as pd 
import numpy as np
from  sklearn import preprocessing

#### ��ȡԭʼ���ݣ�����ȱֵΪNA
def read_raw_data( ):
    '''
    filepath:ԭʼ�ļ�·��
    '''
    
    data1 = pd.read_excel("H:/java/python/src/machinelearning/test/ѵ��.xlsx")
    data2 = pd.read_excel("H:/java/python/src/machinelearning/test/����A.xlsx")
    data3 = pd.read_excel("H:/java/python/src/machinelearning/test/����B.xlsx")
    data = pd.concat([data1,data2,data3],ignore_index=True)
    
#    data1=data1.fillna('NA')
#    data = pd.read_excel("H:/java/python/src/machinelearning/test/trainall.xlsx")
#    data=data.fillna('NA')
#    data.to_excel('H:/java/python/src/machinelearning/test/trainallna.xlsx',index=False)
    return data


#### ��ȡ����R���������,ɾ��NA��
def deal_R_data(filepath):
    '''
    filepath:����KNN�����ļ���·��
    '''
    data = pd.read_csv("imputed_data.csv")
    data = data.dropna(axis=1)
    data = data.ix[:,~(data==0).all()] # ɾ��ֵȫΪ0����
    data = data.astype(dtype=np.string_)
    return data


#### ��׼����ǩ������ǩֵͳһת����range(��ǩֵ����-1)��Χ��
def lebal_encode(data):
    '''
    data:ԭʼ����
    '''
    tool_cols = ['TOOL_ID','Tool','TOOL_ID (#1)','TOOL_ID (#2)','TOOL_ID (#3)',
             'Tool (#1)','Tool (#2)','tool','tool (#1)','TOOL','TOOL (#1)',
             'Tool (#3)','TOOL (#2)','Y']
    le = preprocessing.LabelEncoder() 
    for index in range(len(tool_cols)-1):
        label_encoder=le.fit(data[tool_cols[index]])
        encode_label=label_encoder.transform(data[tool_cols[index]])  
        data[tool_cols[index]] = encode_label
    data.to_excel('H:/java/python/src/machinelearning/test/encode.xlsx',index=False) 
    return data

#### ��ԭʼ��������ȡʱ��
def get_date( ):
    '''
    data :����KNN���������,notice ת����TOOL_ID����
    '''
    data1 = pd.read_excel("H:/java/python/src/machinelearning/test/ѵ��.xlsx")
    data2 = pd.read_excel("H:/java/python/src/machinelearning/test/����A.xlsx")
    data3 = pd.read_excel("H:/java/python/src/machinelearning/test/����B.xlsx")
    data = pd.concat([data1,data2,data3],ignore_index=True)
    data=data.fillna('NA')
    
    
    feature_cols = data.columns
    tool_cols = ['TOOL_ID','Tool','TOOL_ID (#1)','TOOL_ID (#2)','TOOL_ID (#3)',
             'Tool (#1)','Tool (#2)','tool','tool (#1)','TOOL','TOOL (#1)',
             'Tool (#3)','TOOL (#2)','Y']
    date_cols = [ ] # store date data
    '''
    get index of tool from data
    '''

    tool_index_from_feature_cols = [ ]
    
    for i in range(len(tool_cols)):
        for j in range(len(feature_cols)):
            if(tool_cols[i]==feature_cols[j]):
                tool_index_from_feature_cols.append(j)
    
    print(tool_index_from_feature_cols)
    '''
            #get date according to tool_id 
    '''
    
    rule =  r'^2017|2016'
    compiled_rule = re.compile(rule)
    
    '''
                        ��ÿ��Tool_IDд��ʱ��
    '''
    feture_index = tool_index_from_feature_cols[0]
    for tool_index in range(len(tool_cols)-1):
        while(feture_index>=tool_index_from_feature_cols[tool_index] and 
              feture_index<=tool_index_from_feature_cols[tool_index+1]):
                if( compiled_rule.findall(str(data.ix[0,feature_cols[feture_index]]))==['2017'] or 
                    compiled_rule.findall(str(data.ix[0,feature_cols[feture_index]]))==['2016']):
                    date_cols.append(feature_cols[feture_index]) 
                feture_index = feture_index +1
        feture_index = tool_index_from_feature_cols[tool_index+1]
        
        '''
            write to *.csv
        '''                      
        data[date_cols].to_csv('H:\\java\\python\\src\\machinelearning\\date\\extract_'+tool_cols[tool_index]
                              +str(tool_index)+'.csv',index=False)
        data=data.drop(date_cols,axis=1)
        date_cols.clear()
#        date = data[date_cols]
#        
#    '''
#                            �Ȱѿ�ȱ������
#    '''
#    diff_date = (date.max(axis=1)) - (date.min(axis=1)) # ��raw_data�лز���һ��,ָ���ض���λ��
#    diff_date_column = 'diff_date_' + str(tool_index+1)
#    col_name = data.columns.tolist()
#    col_name.insert(col_name.index(tool_cols[tool_index+1]),diff_date_column)# �� �ض���ǰ�����diff_date_column
#    data[diff_date_column] = diff_date   # ����
#    
#    date_cols.clear()
    
#    data.to_csv("fill_diff_date.csv",index=False)

    le = preprocessing.LabelEncoder() 
    for index in range(len(tool_cols)-1):
        label_encoder=le.fit(data[tool_cols[index]])
        encode_label=label_encoder.transform(data[tool_cols[index]])  
        data[tool_cols[index]] = encode_label
    data.to_excel('H:/java/python/src/machinelearning/test/encode.xlsx',index=False) 
    print('ok')
    return data
    


'''
test
''' 


#get_date( )
get_date( )










































































































































    
#'''
#ȫ��ʱ��һ��д��
#'''
#print('ok')
#for index1 in range(size[1]):
#    if( compiled_rule.findall(str(data.ix[0,feature_cols[index1]]))==['2017'] or 
#                compiled_rule.findall(str(data.ix[0,feature_cols[index1]]))==['2016']):
#                date_cols.append(feature_cols[index1])
#date = data[date_cols]
#date.to_csv('extract.csv',index=False)