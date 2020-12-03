from sklearn import metrics
y_true = [0,1,0,1,1]
y_pred_prob= [0.1,0.5,0.6,0.7,0.8]
import matplotlib.pyplot as plt
false_positive_rate,true_positive_rate,_=metrics.roc_curve(y_true,y_pred_prob,pos_label=1)
roc_auc=metrics.auc(false_positive_rate, true_positive_rate)
print(false_positive_rate,true_positive_rate,_)
plt.plot(false_positive_rate,true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
plt.xlim(0,1.05)
plt.ylim(0,1.05)
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()
print('roc_auc_score:',metrics.roc_auc_score(y_true,y_pred_prob))
#[0.  0.  0.5 0.5 1. ] [0.33333333 0.66666667 0.66666667 1.         1.        ] [0.8 0.7 0.6 0.5 0.1]
#threshold=0.6时，取样[0,1,[0,1,1]],假正率为预测的一个0/一共2个0=0.5,真正率为预测的2个1/一共3个1=0.67