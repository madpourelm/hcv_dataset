#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


hcv=pd.read_csv("/Users/habibe/Desktop/Machine learning/Dataset/csv/hcvdat0-1.csv")
hcv


# In[ ]:


hcv.drop("Unnamed: 0" , axis=1 , inplace=True)
hcv


# In[ ]:


hcv.head(10)


# In[ ]:


hcv.tail(10)


# In[ ]:


# duplicates


# In[ ]:


hcv.duplicated().sum()


# In[ ]:


# Missing values


# In[ ]:


hcv.info()


# In[ ]:


hcv.isnull().sum()


# In[ ]:


# first method


# In[ ]:


# Example : "ALP"


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(hcv.ALP.loc[0:25].index , hcv.ALP.loc[0:25].values)
plt.show()


# In[ ]:


plt.hist(hcv.ALP)
plt.show()


# In[ ]:


hcv.ALP.describe()


# In[ ]:


IQR_ALP=80.1-52.5
IQR_ALP


# In[ ]:


upper_extreme=80.1+(1.5 * IQR_ALP)
upper_extreme


# In[ ]:


lower_extreme=52.5-(1.5 * IQR_ALP)
lower_extreme


# In[ ]:


(hcv.ALP[hcv.ALP>121.49999999999999]).sort_values()


# In[ ]:


hcv.ALP[hcv.ALP<11.100000000000009]


# In[ ]:


hcv.drop({539 ,606 , 610 } ,  axis=0 , inplace=True )


# In[ ]:


plt.hist(hcv.ALP);


# In[ ]:


hcv.shape


# In[ ]:


# second method


# In[ ]:


# pairplot 


# In[ ]:


import seaborn as sb


# In[ ]:


#sb.pairplot(hcv)
#plt.show()


# ![Unknown--------.png](attachment:Unknown--------.png)

# In[ ]:


# Normalize


# In[ ]:


# get_dumies


# In[ ]:


# y , x1


# In[ ]:


y=hcv.Category


# In[ ]:


x1=hcv.loc[ : , (hcv.columns!="Category")]
x1


# In[ ]:


x1=pd.get_dummies(x1)
x1


# In[ ]:


x1.describe()


# In[ ]:


plt.figure(figsize=(9 , 5 ))
plt.bar(x1.columns , x1.var());


# In[ ]:


# Scale


# In[ ]:


from sklearn.preprocessing import scale


# In[ ]:


scale(x1)


# In[ ]:


x=pd.DataFrame(scale(x1) , index=x1.index , columns=x1.columns)
x


# In[ ]:


x.boxplot(figsize=(9 ,5));


# In[ ]:


x.values[x.values>5]


# In[ ]:


x.values[x.values<-5]


# In[ ]:


x[x>5].notnull().sum()


# In[ ]:


x[x<-5].notnull().sum()


# In[ ]:


x.ALB[x.ALB>5]


# In[ ]:


x.ALB[x.ALT>5]


# In[ ]:


x.ALB[x.AST>5]


# In[ ]:


x.ALB[x.BIL>5]


# In[ ]:


x.ALB[x.CREA>5]


# In[ ]:


x.ALB[x.GGT>5]


# In[ ]:


x.ALB[x.PROT<-5]


# In[ ]:


outliers=np.array([216 , 536 , 570 , 582 , 592 , 558 , 588 ,
              595 , 609 , 587 , 590 , 597 , 600 , 601 , 
              605 , 586 , 591 , 533 , 558 , 559 , 593 , 602 , 535])


# In[ ]:


outliers=np.unique(outliers)


# In[ ]:


# remove outliers


# In[ ]:


x.drop( outliers , axis=0 , inplace=True)


# In[ ]:


y.drop(outliers ,inplace=True )


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


x.boxplot(figsize=(9 ,5));


# In[ ]:


x.sort_index(ascending=False  , ignore_index=True , sort_remaining=True)


# In[ ]:


# or more simple


# In[ ]:


x=pd.DataFrame(x.values , index=np.arange(0 , 590) , columns=x.columns)
x


# In[ ]:


# strategy="mean" 


# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


imp=SimpleImputer(missing_values=np.nan , strategy="mean" )


# In[ ]:


imp.fit_transform(x)


# In[ ]:


x=pd.DataFrame(imp.fit_transform(x) , columns=x.columns)
x


# In[ ]:


x.isnull().sum()


# In[ ]:


# PCA 


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca=PCA()


# In[ ]:


# n_components=?


# In[ ]:


x.var()


# In[ ]:


plt.figure(figsize=(9 , 5 ))
plt.bar(x.columns , x.var());


# In[ ]:


pca=PCA(n_components=8)


# In[ ]:


x=pca.fit_transform(x)


# In[ ]:


# KNN


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier()


# In[ ]:


# n_neighbors=?


# In[ ]:


# train_test_split


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train , x_test , y_train , y_test=train_test_split(x , y , test_size=0.3 , random_state=2 , stratify=y  )


# In[ ]:


# GridSearchCV


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid={"n_neighbors":np.arange(1 , 31)}


# In[ ]:


cv=GridSearchCV(knn ,param_grid , cv=10)
cv


# In[ ]:


cv.fit(x_train , y_train)


# In[ ]:


cv.best_params_


# In[ ]:


cv.best_score_


# In[ ]:


# test score


# In[ ]:


cv.score(x_test , y_test)

