#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('creditcard.csv')


# In[3]:


print(data.columns)


# In[4]:


print(data.shape)


# In[5]:


data.head()


# In[7]:


print(data.describe())


# In[8]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[12]:


from sklearn.preprocessing import StandardScaler


data['normalisedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()


# In[22]:


Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]
Fraud_ratio = len(Fraud)/float(len(Valid))
print("Fruad ratio : ",Fraud_ratio)
Fraud = 'Fraud Cases: {}'.format(len(data[data['Class'] == 1]))
Valid = 'Valid Transactions: {}'.format(len(data[data['Class'] == 0]))


# In[14]:


print(data.shape)


# In[15]:


X = data.iloc[:,:29].values


# In[17]:


Y = data.iloc[:,-1].values


# In[18]:


print(Y.shape)


# In[23]:


# Number of data points in the minority class
number_records_fraud = len(Fraud)


# In[24]:


fraud_indices = np.array(data[data.Class == 1].index)


# In[27]:


# Number of data points in the majority class
normal_indices = np.array(data[data.Class == 0].index)


# In[28]:


random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)


# In[33]:


# Appending the 2 indices
new_sample_indices = np.concatenate([fraud_indices,random_normal_indices])


# In[34]:


# Under new sample dataset
new_sample_data = data.iloc[new_sample_indices,:]


# In[36]:


X_underNewSample = new_sample_data.iloc[:, new_sample_data.columns != 'Class']
y_underNewSsample = new_sample_data.iloc[:, new_sample_data.columns == 'Class']


# In[37]:


# Showing ratio
print("Percentage of normal transactions: ", len(new_sample_data[new_sample_data.Class == 0])/len(new_sample_data))
print("Percentage of fraud transactions: ", len(new_sample_data[new_sample_data.Class == 1])/len(new_sample_data))
print("Total number of transactions in resampled data: ", len(new_sample_data))


# In[42]:


# Splitting Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))


# In[44]:


# Under new sampled dataset
X_train_underNewsample, X_test_underNewsample, y_train_underNewsample, y_test_underNewsample = train_test_split(X_underNewSample
                                                                                                   ,y_underNewSsample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_underNewsample))
print("Number transactions test dataset: ", len(X_test_underNewsample))
print("Total number of transactions: ", len(X_train_underNewsample)+len(X_test_underNewsample))


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


# In[81]:


def printing_Kfold_scores(x_train_data,y_train_data):
    #fold = KFold(len(y_train_data),False,1) 
    fold = KFold(2,shuffle=False)
    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate (fold.split(x_train_data)):

            #print('Iteration:', i)
            
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())
            # Predict values using the test indices in the training data
            y_pred_underNewsample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_underNewsample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    #best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    best_c = results_table
    best_c.dtypes.eq(object) # you can see the type of best_c
    new = best_c.columns[best_c.dtypes.eq(object)] #get the object column of the best_c
    best_c[new] = best_c[new].apply(pd.to_numeric, errors = 'coerce', axis=0) # change the type of object
    best_c
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter'] #calculate the mean values


    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c
best_c = printing_Kfold_scores(X_train_underNewsample,y_train_underNewsample)


# In[ ]:




