#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


from sklearn.datasets import load_breast_cancer


# In[7]:


dataset=load_breast_cancer()


# In[8]:


dataset 


# In[9]:


x=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)


# In[10]:


x


# In[11]:


y=dataset.target


# In[12]:


## train test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=34)


# In[13]:


x_train


# In[14]:


x_test


# In[15]:


y_train


# In[16]:


y_test


# In[17]:


x_train.shape


# In[18]:


y_train.shape


# In[19]:


x_test.shape


# In[20]:


y_test.shape


# In[21]:


x_train.isnull().sum()


# In[22]:


x_test.isnull().sum()


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


## Post pruning
treeclassifier=DecisionTreeClassifier()


# In[25]:


treeclassifier.fit(x_train,y_train)


# In[26]:


from sklearn import tree

plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)


# In[27]:


## Post pruning with max_depth =2
treeclassifier=DecisionTreeClassifier(max_depth=4)
treeclassifier.fit(x_train,y_train)


# In[28]:


plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)


# In[29]:


## prediction


# In[30]:


y_pred=treeclassifier.predict(x_test)


# In[31]:


y_pred


# In[32]:


from sklearn.metrics import accuracy_score,classification_report


# In[33]:


score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




