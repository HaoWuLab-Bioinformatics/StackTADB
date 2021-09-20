from sklearn.model_selection import cross_val_score,cross_val_predict
import sys
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold

filename = 'dm3.kc167.example.h5'
f = h5py.File(filename, 'r')
x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
x_val = np.array(f['x_val'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])
y_val= np.array(f['y_val'])

a = np.hstack((y_train,y_val))
y = np.hstack((a,y_test))
'''
x1 = np.loadtxt('feature_k-mers_k=6.txt')
x2 = np.loadtxt('feature_NPSE_k=9.txt')
x3 = np.loadtxt('feature_PSSM.txt')
x4 = np.loadtxt('feature_Mismatch_k=3.txt')
x5 = np.loadtxt('feature_subsequence_profile_k=3_delta=0.5.txt')
x6 = np.loadtxt('feature_psednc.txt')
x7 = np.loadtxt('feature_NV_k=1.txt')
'''
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData


x = np.loadtxt('feature_k-mers_k=6.txt')
x = noramlization(x)
'''x1 = np.loadtxt('feature_k-mers_k=6.txt')
x1 = noramlization(x1)
x2 = np.loadtxt('feature_NPSE_k=7.txt')
x2 = noramlization(x2)
x3 = np.loadtxt('feature_Mismatch_k=8.txt')
x3 = noramlization(x3)
x4 = np.loadtxt('feature_subsequence_profile_k=6_delta=0.5.txt')
x4 = noramlization(x4)
x5 = np.loadtxt('feature_PSSM.txt')
x5 = noramlization(x5)
x6 = np.loadtxt('feature_psednc.txt')
x6 = noramlization(x6)
x7 = np.loadtxt('feature_NV_k=1.txt')
x7 = noramlization(x7)'''




print(x.shape)
print('load over')
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)

x_train,_, y_train, _ = train_test_split(x_train,y_train,test_size=0.5, random_state=10)
print(x_train.shape)
print(x_test.shape)
model1 = LogisticRegression(random_state=10, max_iter=50000,penalty='l2', C=1.0)
model2 = KNeighborsClassifier(n_neighbors=1,leaf_size=30, p=2)
model3 = RandomForestClassifier(n_estimators = 300, random_state=10)
model4 = svm.SVC(random_state=10,probability=True,gamma = 'scale',C=1.0,kernel= 'rbf')

stack = StackingCVClassifier(
    classifiers=[model1, model2, model3], meta_classifier=model4, random_state=10,use_probas=True,cv=5)


kf = KFold(10, True, 10)
i=0
acc_stack=np.zeros(10)
aucv=np.zeros(10)
mcc=np.zeros(10)
precision=np.zeros(10)
recall=np.zeros(10)
fscore=np.zeros(10)
for train_index, test_index in kf.split(x_train):
    stack.fit(x_train[train_index], y_train[train_index])
    stack_pred = stack.predict_proba(x_train[test_index])
    stack_predict = stack_pred[:, 1]
    stack_p = stack.predict(x_train[test_index])
    acc_stack[i] = accuracy_score(y_train[test_index], stack_p)

    fpr, tpr, thresholdTest = roc_curve(y_train[test_index], stack_predict)
    aucv[i] = auc(fpr, tpr)
    mcc[i] = matthews_corrcoef(y_train[test_index], stack_p)
    precision[i],recall[i],fscore[i],support=precision_recall_fscore_support(y_train[test_index],stack_p,average='macro')

    i = i + 1
    print('auc:', aucv)
    print('acc:',acc_stack)
    print('mcc:', mcc)
    print('precision:', precision)
    print('recall:', recall)
    print('fscore:', fscore)
print('10-fold cross-validation on the training set')
print('auc:', aucv.mean())
print('acc:',acc_stack.mean())
print('mcc:', mcc.mean())
print('precision:', precision.mean())
print('recall:', recall.mean())
print('fscore:', fscore.mean())
print(aucv.mean(),acc_stack.mean(),mcc.mean(),precision.mean(),recall.mean(),fscore.mean())


stack.fit(x_train, y_train)
stack_pred = stack.predict_proba(x_test)
stack_predict = stack_pred[:, 1]
stack_p = stack.predict(x_test)
acc_stack_test = accuracy_score(y_test, stack_p)

fpr, tpr, thresholdTest = roc_curve(y_test, stack_predict)
aucv_test = auc(fpr, tpr)
mcc_test = matthews_corrcoef(y_test, stack_p)
precision_test, recall_test, fscore_test, support = precision_recall_fscore_support(y_test, stack_p,
                                                                              average='macro')

print('Evaluation on the test set')
print('auc:', aucv_test)
print('acc:',acc_stack_test)
print('mcc:', mcc_test)
print('precision:', precision_test)
print('recall:', recall_test)
print('fscore:', fscore_test)
print(aucv_test,acc_stack_test,mcc_test,precision_test,recall_test,fscore_test)