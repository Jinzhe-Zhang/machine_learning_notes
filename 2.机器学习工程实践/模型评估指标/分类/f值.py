y_true = [1,0,2,0,1,0,2,0,0,2]
y_pred= [1,0,1,0,0,0,2,0,2,1]
from sklearn.metrics import f1_score
print (f1_score(y_true,y_pred,average='weighted',pos_label=1))