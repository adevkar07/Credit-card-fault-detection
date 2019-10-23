#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


# In[4]:


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[5]:


print(data.columns)


# In[6]:


print(data.shape)


# In[7]:


print(data.describe())


# In[12]:


data.iloc[:,-1].hist()
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


# In[82]:


Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]
Fraud_ratio = len(Fraud)/float(len(Valid))
print("Fruad ratio : ",Fraud_ratio)
Fraud = 'Fraud Cases: {}'.format(len(data[data['Class'] == 1]))
Valid = 'Valid Transactions: {}'.format(len(data[data['Class'] == 0]))
print(Fraud)
print(Valid)


# In[83]:


X = data.iloc[:,:30].values


# In[84]:


Y = data.iloc[:,-1].values


# In[70]:


print(Y.shape)


# In[71]:


print("X",X)


# In[72]:


print("Y",Y)


# In[85]:



# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[86]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=Fraud_ratio,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=Fraud_ratio)}


# In[87]:


# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




