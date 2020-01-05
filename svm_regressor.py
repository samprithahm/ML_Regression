# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:18:03 2019

@author: Sampritha H M
"""

# data loading
import pandas as pd
data = pd.read_csv('steelData.csv')

# data pre processing
#Using Pearson Correlation
correlation = data.corr()
#Correlation with output variable
correlation_target = abs(correlation["tensile_strength"])
#Selecting  corelated features which are atleast 10% contributing in prediction
relevant_features = correlation_target[correlation_target>0.1]
# drop the features which do not contribute significantly to predict the target
relevant_features_data = data.drop(['percent_chromium','manufacture_year','percent_nickel','percent_manganese'], axis = 1)

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics 
from math import sqrt
from sklearn.model_selection import train_test_split

# data standardization
scaler = StandardScaler()
X = data.drop(['tensile_strength','sample_id'],axis=1)
X = scaler.fit_transform(X)
y = data['tensile_strength'].values
# split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
# select kernel as linear with penalty parameter C 
regressor = SVR(kernel = 'linear', gamma = 'auto')
# train model with cv of 10
cv_scores = cross_val_score(regressor, X, y, cv=10)

import matplotlib.pyplot as plt
# plot the graph for each fold in knn
cv_range = (1,2,3,4,5,6,7,8,9,10)
plt.figure()
plt.plot(cv_range,cv_scores, '-o', MarkerEdgeColor='c', MarkerFaceColor = 'w', markersize = 10)
plt.xlabel('Number of folds')
plt.ylabel('R2 Scores')
plt.savefig("SVM_CV.png")
print("R2 Score : ", np.mean(cv_scores))

regressor.fit(X_train, y_train) #fit the model
y_pred = regressor.predict(X_test) #make prediction on test set
y2_pred = regressor.predict(X_train)
rmse_val = sqrt(metrics.mean_squared_error(y_test,y_pred)) #calculate rmse
rmse_val2 = sqrt(metrics.mean_squared_error(y_train,y2_pred))
print('RMSE value for test data = ', rmse_val)
print('RMSE value for train data = ', rmse_val2)

import numpy as np
plt.figure()
plt.plot(y_test, y_pred, '*', MarkerEdgeColor='c', MarkerFaceColor = 'w', markersize = 10 )
plt.title('Support Vector Regression Model')
plt.xlabel('Actual Tensile Strength values')
plt.ylabel('Predicted Tensile Strength values')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color = 'red')
plt.show()
plt.savefig("SVM_R2.png")