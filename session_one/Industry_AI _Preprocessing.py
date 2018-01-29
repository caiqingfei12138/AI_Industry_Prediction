# coding=gbk
'''
Created on 2017年12月23日

@author: 14941
'''
import re
import pandas as pd 
import numpy as np
from  sklearn import preprocessing

#### 读取原始数据，填充空缺值为NA
def read_raw_data( ):
    '''
    filepath:原始文件路径
    '''
    
    data1 = pd.read_excel("H:/java/python/src/machinelearning/test/训练.xlsx")
    data2 = pd.read_excel("H:/java/python/src/machinelearning/test/测试A.xlsx")
    data3 = pd.read_excel("H:/java/python/src/machinelearning/test/测试B.xlsx")
    data = pd.concat([data1,data2,data3],ignore_index=True)
    
#    data1=data1.fillna('NA')
#    data = pd.read_excel("H:/java/python/src/machinelearning/test/trainall.xlsx")
#    data=data.fillna('NA')
#    data.to_excel('H:/java/python/src/machinelearning/test/trainallna.xlsx',index=False)
    return data


#### 读取经过R处理的数据,删除NA列
def deal_R_data(filepath):
    '''
    filepath:经过KNN填充后，文件的路径
    '''
    data = pd.read_csv("imputed_data.csv")
    data = data.dropna(axis=1)
    data = data.ix[:,~(data==0).all()] # 删除值全为0的列
    data = data.astype(dtype=np.string_)
    return data


#### 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
def lebal_encode(data):
    '''
    data:原始数据
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

#### 从原始数据中提取时间
def get_date( ):
    '''
    data :经过KNN填充后的数据,notice 转化后TOOL_ID乱码
    '''
    data1 = pd.read_excel("H:/java/python/src/machinelearning/test/训练.xlsx")
    data2 = pd.read_excel("H:/java/python/src/machinelearning/test/测试A.xlsx")
    data3 = pd.read_excel("H:/java/python/src/machinelearning/test/测试B.xlsx")
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
                        按每个Tool_ID写出时间
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
#                            先把空缺的填了
#    '''
#    diff_date = (date.max(axis=1)) - (date.min(axis=1)) # 往raw_data中回插入一列,指定特定的位置
#    diff_date_column = 'diff_date_' + str(tool_index+1)
#    col_name = data.columns.tolist()
#    col_name.insert(col_name.index(tool_cols[tool_index+1]),diff_date_column)# 在 特定列前面插入diff_date_column
#    data[diff_date_column] = diff_date   # 插入
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
#全部时间一起写出
#'''
#print('ok')
#for index1 in range(size[1]):
#    if( compiled_rule.findall(str(data.ix[0,feature_cols[index1]]))==['2017'] or 
#                compiled_rule.findall(str(data.ix[0,feature_cols[index1]]))==['2016']):
#                date_cols.append(feature_cols[index1])
#date = data[date_cols]
#date.to_csv('extract.csv',index=False)