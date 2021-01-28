#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


hcv=pd.read_csv("/Users/habibe/Desktop/Machine learning/Dataset/csv/hcvdat0-1.csv")
hcv


# In[3]:


hcv.drop("Unnamed: 0" , axis=1 , inplace=True)
hcv


# In[4]:


hcv.head(10)


# In[5]:


hcv.tail(10)


# In[6]:


# duplicates


# In[7]:


hcv.duplicated().sum()


# In[8]:


# Missing values


# In[9]:


hcv.info()


# In[10]:


hcv.isnull().sum()


# In[11]:


# first method


# In[12]:


# Example : "ALP"


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


plt.plot(hcv.ALP.loc[0:25].index , hcv.ALP.loc[0:25].values)
plt.show()


# In[15]:


plt.hist(hcv.ALP)
plt.show()


# In[16]:


hcv.ALP.describe()


# In[17]:


IQR_ALP=80.1-52.5
IQR_ALP


# In[18]:


upper_extreme=80.1+(1.5 * IQR_ALP)
upper_extreme


# In[19]:


lower_extreme=52.5-(1.5 * IQR_ALP)
lower_extreme


# In[20]:


(hcv.ALP[hcv.ALP>121.49999999999999]).sort_values()


# In[21]:


hcv.ALP[hcv.ALP<11.100000000000009]


# In[22]:


hcv.drop({539 ,606 , 610 } ,  axis=0 , inplace=True )


# In[23]:


plt.hist(hcv.ALP);


# In[24]:


hcv.shape


# In[25]:


# second method


# In[26]:


# pairplot 


# In[27]:


import seaborn as sb


# In[28]:


#sb.pairplot(hcv)
#plt.show()


# ![Unknown--------.png](attachment:Unknown--------.png)

# In[29]:


# Normalize


# In[30]:


# get_dumies


# In[31]:


# y , x1


# In[32]:


y=hcv.Category


# In[33]:


x1=hcv.loc[ : , (hcv.columns!="Category")]
x1


# In[34]:


x1=pd.get_dummies(x1)
x1


# In[35]:


x1.describe()


# In[36]:


plt.figure(figsize=(9 , 5 ))
plt.bar(x1.columns , x1.var());


# In[37]:


# Scale


# In[38]:


from sklearn.preprocessing import scale


# In[39]:


scale(x1)


# In[40]:


x=pd.DataFrame(scale(x1) , index=x1.index , columns=x1.columns)
x


# In[41]:


x.boxplot(figsize=(9 ,5));


# In[42]:


x.values[x.values>5]


# In[43]:


x.values[x.values<-5]


# In[44]:


x[x>5].notnull().sum()


# In[45]:


x[x<-5].notnull().sum()


# In[46]:


x.ALB[x.ALB>5]


# In[47]:


x.ALB[x.ALT>5]


# In[48]:


x.ALB[x.AST>5]


# In[49]:


x.ALB[x.BIL>5]


# In[50]:


x.ALB[x.CREA>5]


# In[51]:


x.ALB[x.GGT>5]


# In[52]:


x.ALB[x.PROT<-5]


# In[53]:


outliers=np.array([216 , 536 , 570 , 582 , 592 , 558 , 588 ,
              595 , 609 , 587 , 590 , 597 , 600 , 601 , 
              605 , 586 , 591 , 533 , 558 , 559 , 593 , 602 , 535])


# In[54]:


outliers=np.unique(outliers)


# In[55]:


# remove outliers


# In[56]:


x.drop( outliers , axis=0 , inplace=True)


# In[57]:


y.drop(outliers ,inplace=True )


# In[58]:


x.shape


# In[59]:


y.shape


# In[60]:


x.boxplot(figsize=(9 ,5));


# In[61]:


x.sort_index(ascending=False  , ignore_index=True , sort_remaining=True)


# In[62]:


# or more simple


# In[63]:


x=pd.DataFrame(x.values , index=np.arange(0 , 590) , columns=x.columns)
x


# In[64]:


# strategy="mean" 


# In[65]:


from sklearn.impute import SimpleImputer


# In[66]:


imp=SimpleImputer(missing_values=np.nan , strategy="mean" )


# In[67]:


imp.fit_transform(x)


# In[68]:


x=pd.DataFrame(imp.fit_transform(x) , columns=x.columns)
x


# In[69]:


x.isnull().sum()


# In[70]:


# PCA 


# In[71]:


from sklearn.decomposition import PCA


# In[72]:


pca=PCA()


# In[73]:


# n_components=?


# In[74]:


x.var()


# In[75]:


plt.figure(figsize=(9 , 5 ))
plt.bar(x.columns , x.var());


# In[76]:


pca=PCA(n_components=8)


# In[77]:


x=pca.fit_transform(x)


# In[78]:


# KNN


# In[79]:


from sklearn.neighbors import KNeighborsClassifier


# In[80]:


knn=KNeighborsClassifier()


# In[81]:


# n_neighbors=?


# In[82]:


# train_test_split


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


x_train , x_test , y_train , y_test=train_test_split(x , y , test_size=0.3 , random_state=2 , stratify=y  )


# In[85]:


# GridSearchCV


# In[86]:


from sklearn.model_selection import GridSearchCV


# In[87]:


param_grid={"n_neighbors":np.arange(1 , 31)}


# In[88]:


cv=GridSearchCV(knn ,param_grid , cv=10)
cv


# In[89]:


cv.fit(x_train , y_train)


# In[90]:


cv.best_params_


# In[91]:


cv.best_score_


# In[92]:


# test score


# In[93]:


cv.score(x_test , y_test)


