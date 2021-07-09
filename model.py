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
x = np.loadtxt('k-mer_array_k=6.txt')
a = np.hstack((y_train,y_val))
y = np.hstack((a,y_test))
print('load over')


model1 = LogisticRegression(random_state=10, max_iter=5000)
model2 = KNeighborsClassifier(n_neighbors=1)
model3 = RandomForestClassifier(n_estimators = 300, random_state=10)
model4 = svm.SVC(random_state=10,probability=True)

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
for train_index, test_index in kf.split(x):
    stack.fit(x[train_index], y[train_index])
    stack_pred = stack.predict_proba(x[test_index])
    stack_predict = stack_pred[:, 1]
    stack_p = stack.predict(x[test_index])
    acc_stack[i] = accuracy_score(y[test_index], stack_p)

    fpr, tpr, thresholdTest = roc_curve(y[test_index], stack_predict)
    aucv[i] = auc(fpr, tpr)
    mcc[i] = matthews_corrcoef(y[test_index], stack_p)
    precision[i],recall[i],fscore[i],support=precision_recall_fscore_support(y[test_index],stack_p,average='macro')

    i = i + 1
    print('auc:', aucv)
    print('acc:',acc_stack)
    print('mcc:', mcc)
    print('precision:', precision)
    print('recall:', recall)
    print('fscore:', fscore)

print('auc:', aucv.mean())
print('acc:',acc_stack.mean())
print('mcc:', mcc.mean())
print('precision:', precision.mean())
print('recall:', recall.mean())
print('fscore:', fscore.mean())
