# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:36:49 2019

@author: Sampritha H M
"""

# data loading
import pandas as pd
data = pd.read_csv('./steelData.csv')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

from sklearn import neighbors
from sklearn import metrics
from math import sqrt

# data pre processing
from sklearn.preprocessing import MinMaxScaler
#Using Pearson Correlation
plt.figure(figsize=(12,10))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
correlation_target = abs(correlation["tensile_strength"])
#Selecting  corelated features which are atleast 10% contributing in prediction
relevant_features = correlation_target[correlation_target>0.1]
# drop the features which do not contribute significantly to predict the target
relevant_features_data = data.drop(['percent_chromium','manufacture_year','percent_nickel','percent_manganese'], axis = 1)
X = relevant_features_data.drop(['tensile_strength','sample_id'],axis=1)

# data standardization
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = relevant_features_data['tensile_strength'].values
# split data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# calculate error rate for each k
r2_score_val = []
rmse_val = [] #to store rmse values for different k
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    rmse = sqrt(metrics.mean_squared_error(y_test,pred)) #calculate rmse
    r2_score = metrics.r2_score(y_test, pred)
    rmse_val.append(rmse) #store rmse values
    r2_score_val.append(r2_score)
    
    """print('For k = ',K,':\n')
    print('RMSE value is: ', rmse)
    print('R2 Error: ',r2_score,'\n')"""

print("Average RMSE on test data : ", np.mean(rmse_val))

rmse_val2 = []
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_train) #make prediction on test set
    rmse2 = sqrt(metrics.mean_squared_error(y_train,y_pred)) #calculate rmse
    rmse_val2.append(rmse2) #store rmse values
    """print('For k = ',K,':\n')
    print('RMSE value is: ', rmse)"""

print("Average RMSE on train data : ", np.mean(rmse_val2))

print("Average R2 Score : ", np.mean(r2_score))

# plot rmse graph
k = range(10)
curve = pd.DataFrame(rmse_val) #elbow curve 
plt.plot(k,curve, '-o', MarkerEdgeColor = 'red', linestyle = '-.', MarkerFaceColor = 'w', markersize=10, label = 'RMSE')
plt.xlabel("K values")
plt.title("Root Mean Sqaured Error")
plt.legend(loc='best')
plt.savefig("KNN_RMSE.png")

reg = neighbors.KNeighborsRegressor(3)
reg.fit(X_train, y_train)
plt.figure()
plt.plot(y_test,reg.predict(X_test), 'o', MarkerEdgeColor = 'blue',  MarkerFaceColor = 'w', markersize=5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, pred, 1))(np.unique(y_test)), color = 'red')
plt.text(250, 100, 'R-squared = %0.2f' % np.mean(r2_score))
plt.show()
plt.savefig("KNN_R2.png")