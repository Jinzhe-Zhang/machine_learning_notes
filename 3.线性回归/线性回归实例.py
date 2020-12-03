#1.波士顿房价数据
from sklearn.datasets import load_boston
boston = load_boston()
X=boston.data
Y=boston.target
print(X.shape)
print(Y.shape)

#2.划分数据集
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7)

#3.数据标准化
from sklearn import preprocessing
standard_X = preprocessing.StandardScaler()
X_train = standard_X.fit_transform(X_train)
X_test = standard_X.transform(X_test)
standard_Y = preprocessing.StandardScaler()
Y_train=standard_Y.fit_transform(Y_train.reshape(-1,1))
Y_test=standard_Y.transform(Y_test.reshape(-1,1))

#4. 运用ElasticNet回归模型训练和预测
from sklearn.linear_model import ElasticNet
ElasticNet_clf=ElasticNet(alpha=0.05,l1_ratio=0.71)
ElasticNet_clf.fit(X_train,Y_train.ravel())
ElasticNet_clf_score=ElasticNet_clf.score(X_test,Y_test.ravel())
print ('ElasticNet模型得分：',ElasticNet_clf_score)
print ('特征权重：',ElasticNet_clf.coef_)
print ('偏置值：',ElasticNet_clf.intercept_)
print ('迭代次数：',ElasticNet_clf.n_iter_)

#5. 画图
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,3))
axes=fig.add_subplot(1,1,1)
line1,=axes.plot(range(len(Y_test)),Y_test,'b',label='Actual_Value')
ElasticNet_clf_result=ElasticNet_clf.predict(X_test)
line2,=axes.plot(range(len(ElasticNet_clf_result)),ElasticNet_clf_result,'r--',label='ElasticNet_Predicted',linewidth=2)
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1,line2])
plt.title('ElasticNet')
plt.show()