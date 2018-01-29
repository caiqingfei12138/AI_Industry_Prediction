#coding=utf-8

import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import ensemble, linear_model
from sklearn.linear_model.base import LinearRegression
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.svm.classes import SVR
from sklearn.tree.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble.bagging import BaggingRegressor

'''
-------------------------------------------------------------------------------------------------
'''



def splitDataset(data,index):
    '''
    Function: split dataset according to tool_id
    data: raw data through simple dealing (including delete zero or delete same valve) 
    index: index od tool_id in data
    '''
    
    feature_cols = data.columns
    
    tool_cols = ['TOOL_ID','Tool','TOOL_ID (#1)','TOOL_ID (#2)','TOOL_ID (#3)',
             'Tool (#1)','Tool (#2)','tool','tool (#1)','TOOL','TOOL (#1)',
             'Tool (#3)','TOOL (#2)','Y']
   
    '''
    get index of tool from data
    '''

    tool_index_from_feature_cols = [ ]
    
    for i in range(len(tool_cols)):
        for j in range(len(feature_cols)):
            if(tool_cols[i]==feature_cols[j]):
                tool_index_from_feature_cols.append(j)
    
    train_tool_data = data.ix[0:498,feature_cols[tool_index_from_feature_cols[index]:tool_index_from_feature_cols[index+1]]]
    test_tool_data = data.ix[499:598,feature_cols[tool_index_from_feature_cols[index]:tool_index_from_feature_cols[index+1]]]
#    
    return train_tool_data,test_tool_data 
    
 
'''
-------------------------------------------------------------------------------------------------
'''   
    
def train_all_data(data):
    '''
    Function : train model with all data
    data:raw data through simple dealing (including delete zero or delete same valve) 
    '''
    size = data.shape
    X= data.ix[0:498,0:size[1]-2]
    y= data.ix[0:498,size[1]-1]
    print("y_train_mean: ",y.mean())
    print("y_train_var: ",y.var())
    
    test_A = data.ix[499:598,0:size[1]-2]
    
   
    return X,test_A ,y


'''
-------------------------------------------------------------------------------------------------
'''

def moudle_select(X,test_A, y,moudelselect,threshold=False,Rate = False):
    
    
    '''
    Function :model
    X : train data 
    test_A : predict data
    y : result label
    predict_A : predict data
    moudelselect : waht' model do you select?
    threshold:False
    Rate:False
    
    
    modelselect :
    1,XGBRegressor
    2,ensemble.RandomForestRegressor
    3,linear_model.Lasso
    4,LinearRegression
    5,linear_model.BayesianRidge
    6,DecisionTreeRegressor
    7,ensemble.RandomForestRegressor
    8,ensemble.GradientBoostingRegressor
    9,ensemble.AdaBoostRegressor
    10,BaggingRegressor
    11,ExtraTreeRegressor
    12,SVR
    13,MLPRegressor
    other:MLPRegressor
    '''
    
    mse = []
    sum_mse = 0.0
    predict_A = pd.DataFrame(np.zeros((100,10)))
    
    for index in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y) 
        
        
        if(moudelselect==1):
            model = xgb.XGBRegressor(model = xgb.XGBRegressor(max_depth=17,
                                     min_child_weigh=5,
                                     eta=0.025,
                                     gamma=0.06,
                                     subsample=1,
                                     learning_rate=0.1, 
                                     n_estimators=100, 
                                     silent=0, 
                                     n_jobs=-1,
                                     objective='reg:linear'))
            
        elif(moudelselect==2):
            model=ensemble.RandomForestRegressor(n_estimators=25,
                                              criterion='mse',
                                              max_depth=14,
                                              min_samples_split=0.1,
                                              min_samples_leaf=2,
                                              min_weight_fraction_leaf=0.0,
                                              max_features=0.95,
                                              max_leaf_nodes=None,
                                              min_impurity_split=1e-07,
                                              bootstrap=True,
                                              oob_score=False,
                                              n_jobs=-1,
                                              random_state=None,
                                              verbose=0,
                                              warm_start=False)
        elif(moudelselect==3):
            model = linear_model.Lasso(alpha = 0.1,max_iter=1000,normalize=False)
        
        elif(moudelselect==4):
            model=LinearRegression(fit_intercept=False, n_jobs=1, normalize=False)
        
        elif(moudelselect==5):
            model = linear_model.BayesianRidge(alpha_1=1e-06, 
                                    alpha_2=1e-06, 
                                    compute_score=False, 
                                    copy_X=True,
                                    fit_intercept=True, 
                                    lambda_1=1e-06, 
                                    lambda_2=1e-06, 
                                    n_iter=500,
                                    normalize=False,
                                    tol=10, 
                                    verbose=False) 
        
        elif(moudelselect==6):
            model=DecisionTreeRegressor(criterion='mse',splitter='best', 
                             max_depth=3, 
                             min_samples_split=0.1, 
                             min_samples_leaf=0.1, 
                             min_weight_fraction_leaf=0.1, 
                             max_features=None, 
                             random_state=None, 
                             max_leaf_nodes=None, 
                             presort=False)
        
        elif(moudelselect==7):
            model=ensemble.RandomForestRegressor(n_estimators=1000,
                                      criterion='mse',
                                      max_depth=14,
                                      min_samples_split=0.1,
                                      min_samples_leaf=2,
                                      min_weight_fraction_leaf=0.0,
                                      max_features='auto',
                                      max_leaf_nodes=None,
                                      min_impurity_split=1e-07,
                                      bootstrap=True,
                                      oob_score=False,
                                      n_jobs=-1,
                                      random_state=None,
                                      verbose=0,
                                      warm_start=False)
        elif(moudelselect==8):
            model = ensemble.GradientBoostingRegressor(n_estimators=800, 
                                            learning_rate=0.1,
                                            max_depth=4, 
                                            random_state=0, 
                                            loss='ls')
        
        elif(moudelselect==9):
            model= ensemble.AdaBoostRegressor(base_estimator=None, 
                                   n_estimators=120, 
                                   learning_rate=1, 
                                   loss='linear', 
                                   random_state=None)
        
        elif(moudelselect==10):
            model=BaggingRegressor(base_estimator=None, n_estimators=500, 
                         max_samples=1.0, max_features=1.0, 
                         bootstrap=True)
        elif(moudelselect==11):
            model=ExtraTreeRegressor(criterion='mse', splitter='random',                                                             
                                       max_depth=3,
                                       min_samples_split=0.1,                        
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.01,
                                       max_features='auto',
                                       random_state=None,
                                       max_leaf_nodes=None,
                                       min_impurity_split=1e-07
                                       )
  
        elif(moudelselect==12):               
            model=SVR(kernel='rbf', 
            degree=3, 
            gamma='auto', 
            coef0=0.1, 
            tol=0.001, 
            C=1, 
            epsilon=0.1, 
            shrinking=True, 
            cache_size=200, 
            verbose=False, 
            max_iter=-1)
        
        elif(moudelselect==13):
            model=MLPRegressor(hidden_layer_sizes=(100, ), 
                                            activation='relu', solver='adam', alpha=0.0001, 
                                            batch_size='auto', learning_rate='constant', 
                                            learning_rate_init=0.001, power_t=0.5, 
                                            max_iter=200, shuffle=True, 
                                            random_state=None, tol=0.0001, verbose=False, 
                                            warm_start=False, momentum=0.9, 
                                            nesterovs_momentum=True, early_stopping=False, 
                                            validation_fraction=0.1, beta_1=0.9, 
                                            beta_2=0.999, epsilon=1e-08)     
        else:
            model=MLPRegressor(activation='relu',alpha=0.001,
                     solver='lbfgs', max_iter=90,
                     hidden_layer_sizes=(11, 11, 11),
                     random_state=1)
    
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("index: ",index,mean_squared_error(y_test,y_pred))
        sum_mse += mean_squared_error(y_test,y_pred)
#       
#        
        if(threshold==False):
            y_predict = model.predict(test_A)
            predict_A.ix[:,index] = y_predict 
            mse.append(mean_squared_error(y_test,y_pred))
        else:
            if(mean_squared_error(y_test,y_pred)<=0.03000):
                y_predict = model.predict(test_A)
                predict_A.ix[:,index] = y_predict 
                mse.append(mean_squared_error(y_test,y_pred))
    
    
    
     
#        if(Rate==False):
#            mse_rate = mse / np.sum(mse)
#            #predict_A = predict_A.ix[:,~(data==0).all()] 
#            for index in range(len(mse_rate)):
#                y+=predict_A.ix[:,index]*mse_rate[index]
#     
    y=0.0
    mse = mse / np.sum(mse)
    mse = pd.Series(mse)
    mse_rate_asc = mse.sort_values(ascending=False)
    mse_rate_asc = mse_rate_asc.reset_index(drop=True) 
    mse_rate_desc = mse.sort_values(ascending=True)
    indexs=list(mse_rate_desc.index)                  
    for index in range(len(mse)):
        y+= mse_rate_asc.ix[index]*predict_A.ix[:,indexs[index]]
                
    print("y_predict_mean: ",y.mean())
    print("y_predict_var: ",y.var())
    y=pd.DataFrame(y)
    y.to_excel("H:/java/python/src/machinelearning/test/predict.xlsx",index=False)
    predict_A.to_excel("H:/java/python/src/machinelearning/test/predict_testA.xlsx",index=False)
    print("Averge mse:",sum_mse/len(mse))


'''
-------------------------------------------------------------------------------------------------
'''

def train_spilt_data(data,index,modelselect,y):
    train_tool_data,test_tool_data = splitDataset(data, index)
    X_train, X_test, y_train, y_test = train_test_split(train_tool_data, y) 
  
    moudle_select( train_tool_data, test_tool_data, y, moudelselect=modelselect, threshold=False, Rate=False)
    
def train_all (data,moudelselect):
    X,test_A ,y = train_all_data(data)
    moudle_select(X,test_A, y,moudelselect,threshold=False,Rate = False)

# read data
data = pd.read_csv('H:/java/python/src/machinelearning/test/imputed_data.csv')
data = data.dropna(axis=1) 
data = data.ix[:,~(data==0).all()] 
size = data.shape
cols = data.columns
y= data.ix[0:498,size[1]-1]

for i in range(size[1]):
    if(len(data.ix[:,cols[i]].unique())==1):
        del data[cols[i]]

# 按工序来
#train_spilt_data(data,12,1,y)

# 不同方法
train_all(data,9)