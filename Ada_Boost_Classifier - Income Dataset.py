# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:18:13 2024

@author: Priyanka
"""

#import necessary packages
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_section import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#read CSV file
loan_data=pd.read_csv("C:\Data Set\income1.csv")
loan_data.columns
loan_data.head()
#let us split the data in input and output
x=loan_data.iloc[:,0:6]
y=loan_data.iloc[:,6]

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learining_rate=1)

#n_estimators =number of week learners
#learing rate ,it contributes weights of week learners,bydefault is
#train the model
model=ada_model.fit(x_train,y_train)
#predict the result
y_pred=model.predict(x_test)
print("accuracy",metrics.accuracy_score(y_test,y_pred))

#let us tey for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#here base model is canged
Ada_model=AdaBoostClassifier(n_estimators=50,base_esimator=lr,learining_rate=1)
model=ada_model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy",metrics.accuracy_score(y_test,y_pred))

