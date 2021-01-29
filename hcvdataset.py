
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

# Normalize

# get_dumies

# y , x1

y=hcv.Category

x1=hcv.loc[ : , (hcv.columns!="Category")]
x1

x1=pd.get_dummies(x1)
x1

x1.describe()

plt.figure(figsize=(9 , 5 ))
plt.bar(x1.columns , x1.var());

# Scale

from sklearn.preprocessing import scale

scale(x1)

x=pd.DataFrame(scale(x1) , index=x1.index , columns=x1.columns)
x

x.boxplot(figsize=(9 ,5));

x.values[x.values>5]

x.values[x.values<-5]

x[x>5].notnull().sum()

x[x<-5].notnull().sum()

x.ALB[x.ALB>5]

x.ALB[x.ALT>5]

x.ALB[x.AST>5]

x.ALB[x.BIL>5]

x.ALB[x.CREA>5]

x.ALB[x.GGT>5]

x.ALB[x.PROT<-5]

outliers=np.array([216 , 536 , 570 , 582 , 592 , 558 , 588 ,
              595 , 609 , 587 , 590 , 597 , 600 , 601 , 
              605 , 586 , 591 , 533 , 558 , 559 , 593 , 602 , 535])

outliers=np.unique(outliers)

# remove outliers

x.drop( outliers , axis=0 , inplace=True)

y.drop(outliers ,inplace=True )

x.shape

y.shape

x.boxplot(figsize=(9 ,5));

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

# PCA 

from sklearn.decomposition import PCA

pca=PCA()

# n_components=?

x.var()

plt.figure(figsize=(9 , 5 ))
plt.bar(x.columns , x.var());

pca=PCA(n_components=8)

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


