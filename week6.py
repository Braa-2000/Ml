#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:





# In[80]:


#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=1)


# In[ ]:





# In[81]:


# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="bad",alpha= 0.5)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="good",alpha= 0.5)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor


# In[83]:


# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[84]:


# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[85]:


# %% SVM

from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  


# In[86]:


# %% test
print("print accuracy of naive bayes algo: ",classifier.score(x_test,y_test))


# In[87]:


y_pred = classifier.predict(x_test)  
y_pred


# In[ ]:





# In[ ]:




