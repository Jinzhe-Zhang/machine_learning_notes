import pandas as pd
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
from sklearn.ensemble import RandomForestRegressor
age_df=data_train[['Age','Fare','Parch','SibSp','Pclass']]
age_know=age_df[age_df.Age.notnull()].values
age_unknow=age_df[age_df.Age.isnull()].values
X=age_know[:,1:]
y=age_know[:,0]
print (y.inspect)
RF_clf=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
RF_clf.fit(X,y)
age_predicted=RF_clf.predict(age_unknow[:,1::])
data_train.loc[(data_train.Age.isnull()),'Age']=age_predicted
data_train.info()
data_train=data_train.drop(['Cabin'],axis=1)
data_train.info()
data_train=data_train.dropna(axis=0)
data_train.reset_index(drop=True, inplace=True)
data_train.info()
cate_df=data_train[['Pclass','Sex','Embarked']]
cate_onehot_df=pd.get_dummies(cate_df)
print (cate_onehot_df.head(3))
cont_df=data_train[['Age','Fare','SibSp','Parch']]
import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
temp_scale=scaler.fit(cont_df)
cont_df=pd.DataFrame(scaler.fit_transform(cont_df,temp_scale),columns=['Age_scaled','Fare_Scaled','SibSp_Scaled','Parch_Scaled'])
print (cont_df.head(3))
df_train=pd.concat([data_train['Survived'],cate_onehot_df,cont_df],axis=1)
from sklearn import linear_model
df_train_mat=df_train.as_matrix()
X=df_train_mat[:,1:]
y=df_train_mat[:,0]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
LR_clf=linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
LR_clf.fit(X_train,y_train)
y_predict=LR_clf.predict(X_test)
y_predict_prob=LR_clf.predict_proba(X_test)[:,1]
from sklearn.metrics import classification_report
print ('查准率、查全率、F1值')
print (classification_report(y_test,y_predict,target_names=None))
from sklearn.metrics import roc_auc_score
print ('AUC值')
print (roc_auc_score(y_test,y_predict_prob))
from sklearn.metrics import confusion_matrix
print('混淆矩阵')
print(confusion_matrix(y_test,y_predict,labels=None))
feature_list = list(df_train.columns[1:])
print(feature_list)
weight_array=LR_clf.coef_
weight = weight_array[0]
print (weight)
import numpy as np
df=pd.DataFrame({'feature':feature_list,'weight':weight})
print(df)