#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[2]:


df=pd.read_csv('lone - loan_train.csv')
df


# In[10]:


sns.heatmap (df.isnull(), yticklabels=False, annot=True)


# In[3]:


df.isnull().sum()


# In[4]:


df['Gender'] = df['Gender'].fillna(method='pad')
df['Self_Employed'] = df['Self_Employed'].fillna(method='pad')


# In[5]:


df['Dependents'] = df['Dependents'].fillna(value=df['Dependents'].mean())
df['Term'] = df['Term'].fillna(value=df['Term'].mean())


# In[6]:


df=df.drop(['Credit_History'], axis=1)
df=df.drop(['Married'], axis=1)


# In[7]:


df.isnull().sum()


# In[8]:


target=df.Status
inputs=df.drop('Status', axis='columns')


# In[9]:


dummies1 = pd.get_dummies(inputs.Gender)
dummies2 = pd.get_dummies(inputs.Education)
dummies3 = pd.get_dummies(inputs.Area)


# In[10]:


dummies1.head (3)


# In[11]:


dummies2.head (3)


# In[12]:


dummies3.head(3)


# In[13]:


inputs = pd.concat([inputs, dummies1,dummies2,dummies3], axis='columns')
inputs.head(3)


# In[14]:


inputs.drop('Gender', axis='columns', inplace=True)
inputs.drop('Education', axis='columns',inplace=True)
inputs.drop('Area', axis='columns',inplace=True)
inputs.head(3)


# In[68]:


plt.figure(figsize=(15,10))
sns.heatmap(inputs.corr(), annot=True)


# In[17]:


sns.relplot(x="Loan_Amount", y="Applicant_Income", height=6,hue="Education",data=df)


# In[11]:


sns.pairplot(df)


# In[15]:


X_train, X_test, y_train, y_test=train_test_split(inputs, target, test_size=0.2)


# In[16]:


X_train


# In[17]:


model = GaussianNB()


# In[21]:


model.fit(X_train, y_train)


# In[22]:


model.score(X_test,y_test)


# In[23]:


y_test[:10]


# In[24]:


model.predict(X_test[:10])


# In[25]:


model.predict_proba(X_test[:10])


# In[ ]:




