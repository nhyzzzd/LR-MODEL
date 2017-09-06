# LR-MODEL
about LR model
import pandas as pd
import numpy as np
# you might need your sklearn module.
data = pd.read_csv('http://www.ats.ucla.edu/stat/data/binary.csv')
print data.head(5)
# print data.tail(5)
print data.describe()
print pd.crosstab
#TODO: add your code below
dummy_ranks = pd.get_dummies(data['rank'], prefix='rank')
print dummy_ranks.head(5)
dummy_ranks = pd.get_dummies(data['rank'], prefix='rank')
print dummy_ranks.head()
cols_to_keep = ['admit', 'gre', 'gpa']
data = data[cols_to_keep].join(dummy_ranks.ix[:, 'rank_2':])
print data.head()
data['intercept'] = 1.0
print data
#TODO: add your code below
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['gre','gpa','rank_2','rank_3','rank_4','intercept']], data['admit'], test_size=0.3, random_state=0)
#TODO: add your code below
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
model = LogisticRegression()
model.fit(X_train, y_train)
#TODO: add your code below
from sklearn import metrics
predicted = model.predict(X_test)
print predicted
probs = model.predict_proba(X_test)
print probs
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
