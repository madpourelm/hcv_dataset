
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


sb.pairplot(hcv)
plt.show()


# In[ ]:


# Standardization


# In[ ]:


# get_dumies


# In[ ]:


# y , x


# In[ ]:


y=hcv.Category


# In[ ]:


x=hcv.loc[ : , (hcv.columns!="Category")]
x


# In[ ]:


x=pd.get_dummies(x)
x


# In[ ]:


x.describe()


# In[ ]:


# Standardization


# In[ ]:


from sklearn.preprocessing import scale


# In[ ]:


scale(x)


# In[ ]:


x1=pd.DataFrame(scale(x) , index=x.index , columns=x.columns)
x1


# In[ ]:


x1.boxplot(figsize=(9 ,5));


# In[ ]:


x1.values[x1.values>5]


# In[ ]:


x1.values[x1.values<-5]


# In[ ]:


x1[x1>5].notnull().sum()


# In[ ]:


x1[x1<-5].notnull().sum()


# In[ ]:


x1.loc[:][x1.values>5].index


# In[ ]:


x1.loc[:][x1.values<-5].index


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


# Normalize


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


normalize(x , norm='l1',  axis=1)


# In[ ]:


x=pd.DataFrame(normalize(x) , columns=x.columns)
x


# In[ ]:


# PCA 


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca=PCA()


# In[ ]:


x=pca.fit_transform(x)


# In[ ]:


# n_components=?


# In[ ]:


n=range(pca.n_components_)


# In[ ]:


v=pca.explained_variance_


# In[ ]:


plt.bar(n , v)
plt.xticks(n);


# In[ ]:


pca=PCA(n_components=10)


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


# In[ ]:


# targets in hcv dataset are not balanced


# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report


# In[ ]:


y_pred=cv.predict(x_test)


# In[ ]:


confusion_matrix(y_test , y_pred)


# In[ ]:


print(classification_report(y_test , y_pred))


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


normalize(confusion_matrix(y_test , y_pred) , norm='l1' , axis=1 )


# In[ ]:


y.value_counts()


# In[ ]:


y.replace("0=Blood Donor" , 0 , inplace=True)


# In[ ]:


y.replace(["3=Cirrhosis" ,"0s=suspect Blood Donor" , "1=Hepatitis" , "2=Fibrosis"] , 1 , inplace=True )


# In[ ]:


# change sample


# In[ ]:


y.loc[534]


# In[ ]:


y.value_counts()


# In[ ]:


x_train , x_test , y_train , y_test=train_test_split(x , y , test_size=0.3 , random_state=2 , stratify=y  )


# In[ ]:


knn=KNeighborsClassifier()


# In[ ]:


# change sample


# In[ ]:


y_test


# In[ ]:


# GridSearchCV


# In[ ]:


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

