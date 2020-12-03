from sklearn.metrics import confusion_matrix
y_true = [1,0,2,0,0,0,2,0,0,2]
y_pred= [1,0,1,0,0,0,2,0,2,1]
print (confusion_matrix(y_true,y_pred,labels=None))