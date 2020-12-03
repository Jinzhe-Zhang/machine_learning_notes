from sklearn import datasets
iris = datasets.load_iris()
X=iris.data[:, :2]
Y=iris.target
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#ffaaaa','#aaffaa','#aaaaff'])
cmap_bold=ListedColormap(['#ff0000','#00ff00','#0000ff'])
from sklearn.tree import DecisionTreeClassifier
DT_clf=DecisionTreeClassifier()
DT_clf.fit(X,Y)
x_min,x_max = X[:,0].min() - 1 ,X[:,0].max() + 1
y_min,y_max = X[:,1].min() - 1 ,X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max, .02),np.arange(y_min,y_max,.02))
Z=DT_clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("decision tree use default parameter")
plt.show()