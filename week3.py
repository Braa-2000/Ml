#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()


# In[90]:


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color="yellow",label="bad",alpha= 0.8)
plt.scatter(B.radius_mean,B.texture_mean,color="blue",label="good",alpha= 0.8)
plt.xlabel("radius mean")
plt.ylabel("texture mean")
plt.legend()
plt.show()


# In[91]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[92]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[93]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.8,random_state=1)


# In[94]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(4,knn.score(x_test,y_test)))


# In[95]:


score_list = []
for each in range(2,12):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(2,12),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[96]:


y_pred = knn2.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)



import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(6,6))
sns.heatmap(cm,annot = True,linewidths=0.1,linecolor="yellow",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

