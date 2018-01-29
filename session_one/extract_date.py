# coding=gbk
'''
Created on 2017年12月25日

@author: 14941
'''
import re
import numpy as np
import pandas as pd

data = pd.read_csv('imputed_data.csv')
tool_cols = ['TOOL_ID','Tool','TOOL_ID (#1)','TOOL_ID (#2)','TOOL_ID (#3)',
             'Tool (#1)','Tool (#2)','tool','tool (#1)','TOOL','TOOL (#1)',
             'Tool (#3)','TOOL (#2)','Y']

data = data[24]
#data = data.astype(dtype=np.long)

rule =  r'^2017|2016'
compiled_rule = re.compile(rule)
for index in range(len(data)):
    if(compiled_rule.findall(str(data[index]))==['2017'] or 
                    compiled_rule.findall(str())==['2016']) :               
                        print("ture")
                        print(data[index])