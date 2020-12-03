import math
from sklearn import metrics
labels_true=[0,0,0,1,1]
labels_pred=[0,0,1,1,2]
abcd=[0,0,0,0]
for i in range(len(labels_true)):
    j=i+1
    while j<len(labels_pred):
        p=0
        if labels_true[i]==labels_true[j]:
            p+=2
        if labels_pred[i]==labels_pred[j]:
            p+=1
        abcd[p]+=1
        j+=1
print (abcd)
SS=abcd[0]
SD=abcd[1]
DS=abcd[2]
DD=abcd[3]
JC=SS/(SS+SD+DS)
print ('JC:',JC)
FMI=math.sqrt(SS/(SS+SD)*SS/(SS+DS))
print ('FMI:',FMI)
RI=(SS+DD)/(SS+DD+DS+SD)
print ('RI:',RI)
ARI=metrics.adjusted_rand_score(labels_true,labels_pred)#？？？？？？？？？？？？？？？？？
print ('ARI:',ARI)
#NMI=信息增益
NMI=metrics.adjusted_mutual_info_score(labels_true,labels_pred)
print ('NMI:',NMI)