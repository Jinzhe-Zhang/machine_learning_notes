y_true = [[0.5,1],[-1,1],[7,-6]]
y_pred=[[0,2],[-1,2],[8,-5]]
from sklearn.metrics import mean_absolute_error
print (mean_absolute_error(y_true,y_pred))#对应位置相减，(1+1+1+1+0.5)/6