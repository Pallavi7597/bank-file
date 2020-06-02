# bank-file
decision tree algo
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv('Bank File.csv',header=0)
data.head()
data.info()
data.corr()
data.shape
data.describe()
col_names = ['Salary', 'AccountBalance', 'CreditScore', 'PaymentPending','Fraud']
data.columns = col_names
feature_cols = ['Salary', 'AccountBalance', 'CreditScore', 'PaymentPending']
Y=data['Fraud']
X=data.drop('Fraud',axis=1)
print(X.head())
print(Y.head())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_pred
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
y_pred_train=clf.predict(X_train)
metrics.accuracy_score(y_pred_train,y_train)
#!pip install graphviz
#!pip install pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT Graph-1.png')
Image(graph.create_png())
clf_prune=DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split = 5,splitter='best')
clf_prune = clf_prune.fit(X_train,y_train)
y_pred = clf_prune.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
y_pred_train = clf_prune.predict(X_train)
metrics.accuracy_score(y_train, y_pred_train)
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
dot_data = export_graphviz(clf_prune, out_file=None,  
                filled=True, rounded=True,
                special_characters=True, 
                feature_names = feature_cols,
                class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('DT Graph-2.png')
Image(graph.create_png())
