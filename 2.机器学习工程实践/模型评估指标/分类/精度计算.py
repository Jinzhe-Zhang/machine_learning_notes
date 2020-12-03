y_true = [1,0,2,0,1,0,2,0,0,2]
y_pred= [1,0,1,0,0,0,2,0,2,1]
from sklearn.metrics import precision_score
print (precision_score(y_true,y_pred,average=None))