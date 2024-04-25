#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


dataset=load_iris()


# In[6]:


print(dataset.DESCR)


# In[7]:


import seaborn as sns


# In[8]:


df=sns.load_dataset("iris")


# In[9]:


df


# In[11]:


dataset.target


# In[13]:


### Independent and dependent feature
x=df.iloc[:,:-1]
y=dataset.target


# In[14]:


x,y


# In[15]:


### train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[16]:


x_train


# In[17]:


x_test


# In[18]:


y_train


# In[19]:


y_test


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


x_train.head()


# In[30]:


## Post pruning with max_depth =2
treeclassifier=DecisionTreeClassifier(max_depth=2)
treeclassifier.fit(x_train,y_train)


# In[31]:


plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)


# In[32]:


## prediction


# In[33]:


y_pred=treeclassifier.predict(x_test)


# In[34]:


y_pred


# In[35]:


from sklearn.metrics import accuracy_score,classification_report


# In[37]:


score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




