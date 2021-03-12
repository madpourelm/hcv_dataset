
import numpy as np
import pandas as pd

hcv=pd.read_csv("/Users/habibe/Desktop/Machine learning/Dataset/csv/hcvdat0-1.csv")
hcv

hcv.drop("Unnamed: 0" , axis=1 , inplace=True)
hcv

hcv.head(10)

hcv.tail(10)

# duplicates

hcv.duplicated().sum()

# Missing values

hcv.info()

hcv.isnull().sum()

# first method

# Example : "ALP"

import matplotlib.pyplot as plt

plt.plot(hcv.ALP.loc[0:25].index , hcv.ALP.loc[0:25].values)
plt.show()

plt.hist(hcv.ALP)
plt.show()

hcv.ALP.describe()

IQR_ALP=80.1-52.5
IQR_ALP

upper_extreme=80.1+(1.5 * IQR_ALP)
upper_extreme

lower_extreme=52.5-(1.5 * IQR_ALP)
lower_extreme

(hcv.ALP[hcv.ALP>121.49999999999999]).sort_values()

hcv.ALP[hcv.ALP<11.100000000000009]

hcv.drop({539 ,606 , 610 } ,  axis=0 , inplace=True )

plt.hist(hcv.ALP);

hcv.shape

# second method

# pairplot 

import seaborn as sb

sb.pairplot(hcv)
plt.show()

# Standardization

# get_dumies

# y , x

y=hcv.Category

x=hcv.loc[ : , (hcv.columns!="Category")]
x

x=pd.get_dummies(x)
x

x.describe()

# Standardization

from sklearn.preprocessing import scale

scale(x)

x1=pd.DataFrame(scale(x) , index=x.index , columns=x.columns)
x1

x1.boxplot(figsize=(9 ,5));

x1.values[x1.values>5]

x1.values[x1.values<-5]

x1[x1>5].notnull().sum()

x1[x1<-5].notnull().sum()

x1.loc[:][x1.values>5].index

x1.loc[:][x1.values<-5].index

outliers=np.array([216 , 536 , 570 , 582 , 592 , 558 , 588 ,
              595 , 609 , 587 , 590 , 597 , 600 , 601 , 
              605 , 586 , 591 , 533 , 558 , 559 , 593 , 602 , 535])

outliers=np.unique(outliers)

# remove outliers

x.drop( outliers , axis=0 , inplace=True)

y.drop(outliers ,inplace=True )

x.shape

y.shape

x.sort_index(ascending=False  , ignore_index=True , sort_remaining=True)

# or more simple

x=pd.DataFrame(x.values , index=np.arange(0 , 590) , columns=x.columns)
x

# strategy="mean" 

from sklearn.impute import SimpleImputer

imp=SimpleImputer(missing_values=np.nan , strategy="mean" )

imp.fit_transform(x)

x=pd.DataFrame(imp.fit_transform(x) , columns=x.columns)
x

x.isnull().sum()

# Normalize

from sklearn.preprocessing import normalize

normalize(x , norm='l1',  axis=1)

x=pd.DataFrame(normalize(x) , columns=x.columns)
x

# PCA 

from sklearn.decomposition import PCA

pca=PCA()

x=pca.fit_transform(x)

# n_components=?

n=range(pca.n_components_)

v=pca.explained_variance_

plt.bar(n , v)
plt.xticks(n);

pca=PCA(n_components=10)

x=pca.fit_transform(x)

# KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

# n_neighbors=?

# train_test_split

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test=train_test_split(x , y , test_size=0.3 , random_state=2 , stratify=y  )

# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid={"n_neighbors":np.arange(1 , 31)}

cv=GridSearchCV(knn ,param_grid , cv=10)
cv

cv.fit(x_train , y_train)

cv.best_params_

cv.best_score_

# test score

cv.score(x_test , y_test)

# targets in hcv dataset are not balanced

from sklearn.metrics import confusion_matrix , classification_report

y_pred=cv.predict(x_test)

confusion_matrix(y_test , y_pred)

print(classification_report(y_test , y_pred))

from sklearn.preprocessing import normalize

normalize(confusion_matrix(y_test , y_pred) , norm='l1' , axis=1 )

y.value_counts()

y.replace("0=Blood Donor" , 0 , inplace=True)

y.replace(["3=Cirrhosis" ,"0s=suspect Blood Donor" , "1=Hepatitis" , "2=Fibrosis"] , 1 , inplace=True )

# change sample

y.loc[534]

y.value_counts()

x_train , x_test , y_train , y_test=train_test_split(x , y , test_size=0.3 , random_state=2 , stratify=y  )

knn=KNeighborsClassifier()

# change sample

y_test

# GridSearchCV

param_grid={"n_neighbors":np.arange(1 , 31)}


# In[ ]:


GridSearchCV(knn , param_grid , cv=10 )


# In[ ]:


cv.best_params_


# In[ ]:


cv.best_score_


# In[ ]:


y_pred=cv.predict(x_test)
y_pred


# In[ ]:


y_pred=pd.DataFrame(y_pred)


# In[ ]:


y_pred.replace("0=Blood Donor" , 0 , inplace=True)


# In[ ]:


y_pred.replace(["3=Cirrhosis" ,"0s=suspect Blood Donor" , "1=Hepatitis" , "2=Fibrosis"], 1 , inplace=True )


# In[ ]:


y_pred.value_counts()


# In[ ]:


confusion_matrix(y_test , y_pred)


# In[ ]:


print(classification_report(y_test , y_pred))


# In[ ]:


# Normalize


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


normalize(confusion_matrix(y_test , y_pred) , norm="l1" , axis=1)


# In[ ]:


# LogisticRegression


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log=LogisticRegression()


# In[ ]:


log.fit(x_train , y_train)


# In[ ]:


log.score(x_test , y_test)


# In[ ]:


y_predict=log.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report


# In[ ]:


confusion_matrix(y_test , y_pred)


# In[ ]:


print(classification_report(y_test , y_pred))


# In[ ]:


from sklearn.metrics import roc_curve


# In[ ]:


y_pred_prob=log.predict_proba(x_test)[: , 1]


# In[ ]:


fpr , tpr , thresholds=roc_curve(y_test , y_pred_prob  )


# In[ ]:


plt.plot(fpr , tpr)
plt.plot([0 , 1] , [0 , 1] , "--")
plt.xlabel("fpr")
plt.ylabel("tpr");


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc_score(y_test , y_pred_prob)

