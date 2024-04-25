#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Decision Tree Regressor Implementation


# In[2]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


## California House Pricing Dataset
from sklearn.datasets import fetch_california_housing
california_df=fetch_california_housing()


# In[4]:


california_df


# In[5]:


## Independent feature


# In[6]:


x=pd.DataFrame(california_df.data,columns=california_df.feature_names)


# In[7]:


## dependent feature
y=california_df.target


# In[8]:


x.head()


# In[9]:


x.head()


# In[10]:


## train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[11]:


x_train


# In[12]:


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()


# In[13]:


regressor.fit(x_train,y_train)


# In[14]:


y_pred=regressor.predict(x_test)


# In[15]:


y_pred


# In[16]:


from sklearn.metrics import r2_score


# In[17]:


score=r2_score(y_pred,y_test)


# In[18]:


score


# In[19]:


## Hyperparameter Turuning
parameter={
    "criterion":["squared_error","friedman_mse","absolute_error","poisson"],
    "splitter":["best","random"],
    "max_depth":[1,2,3,4,5,6,7,8,10,11,12],
    "max_features":["auto","sqrt","log2"]
}
regressor=DecisionTreeRegressor()


# In[20]:


## https://scikit-Learn.org/state/modules/model_evaluation.html
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
regressorcv=GridSearchCV(regressor,param_grid=parameter,cv=5,scoring="neg_mean_squared_error")


# In[21]:


regressorcv.fit(x_train,y_train)


# In[23]:


regressorcv.best_params_


# In[24]:


y_pred=regressorcv.predict(x_test)


# In[26]:


r2_score(y_pred,y_test)


# In[27]:


regressorcv.predict()


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




