# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:20:03 2024

@author: Priyanka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

data=pd.read_csv("C:\Data Set\movies_classification.csv")

#Data information
data.head()
data.info()
data.isna().sum()

#EDA
target=data["Start_Tech_Oscar"]
sns.countplot(x= target, palette='winter')
plt.xlabel("Oscar_Rate");

#Our data is evently distributed. Atleast 200 are there is both choice
plt.figure(figsize= (16, 8))
sns.heatmap(data.corr(), annot=True, cmap='YlGnBu', fmt='.2f')

#Observations:
'''
1) lead_Actor_Rating,Lead_Actress_rating,Director_rating and producer
    

'''
sns.countplot(x='Genre', data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class', fontsize=10);

#Observation:
#here are more chances of getting Oscar in Drama, Comedy and Action gener

sns.countplot(x='3D_available',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class', fontsize=10);
#observation
#it is clear from the plot that if 3D is Available the there is a chance of 
sns.set_context('notebook',font_scale=1.2)
fig,ax=plt.subplots(2,figsize=(20,13))
plt.suptitle("Distribution of Twiteer_hastags and collection based on target variable",fontsize=20)
ax1=sns.histplot(x="Twitter_hastags",data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[0],palette='winter')
ax1.set(xlabel='Twitter_hastags',title="Distribution of Twiteer_hastags and collection based on target variable")
ax2=sns.histplot(x='Collection',data=data,hue='Start_Tech_Oscar',kde=True, ax=ax[1],palette='viridis')

ax2.set(xlabel="Collection",title='Distribution of fare based on target variable')

plt.show()
data.hist(bins=30,figsize=(20,15),color='#005b96');
#As we can see there are outliers in Twitter_hastags,Marketing expense,Time-taken
sns.boxplot(x=data['Twitter_hastags'])
sns.boxplot(x=data['Marketing expense'])
sns.boxplot(x=data['Time_taken'])
sns.boxplot(x=data['Avg_age_actors'])



#write code for winsorizor code
#cheking skewness
skew_df=pd.DataFrame(data.select_dtypes(np.number).columns,columns=['feature'])
skew_df['Skew']=skew_df['feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew']=skew_df['Skew'].apply(abs)
skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x>=0.5 else False)
skew_df

#Total Charges column  is clearly skewed as we also saw in the histogram ,
for column in skew_df.query("Skewed ==True")['feature'].values: 
    data[column]=np.log1p(data[column])
data.head()
#Encoding 
data1=data.copy()
data1=pd.get_dummies(data1)

data1.head()
#Scaling
data2=data1.copy()
sc = StandardScaler()
data2[data1.select_dtypes(np.number).columns]=sc.fit_transform(data2[data1.select_dtype(np.number).columns])
data2.drop(["Start_Tech_Oscar"],axis=1, inplace=True)
data2.head()

#Splitting 
data_f=data.copy()
target=data["Start_Tech_Oscar"]
target=target.astype(int)
target
x_train,x_test, y_train,y_test=train_test_split(data_f,target,test_size=0.2,stratify=target,random_state=42)

#Modeling

from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

#Evaluation on testing data
confusion_matrix(y_test,ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

#Evaluation of traing data
accuracy_score(y_train, ada_clf.predict(x_train))

from sklearn.metrics import accuracy_score,confusion_matrix

#Evaluation on Testing data
