#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Original jupyter notebook from https://github.com/vishabh123/vishabh


# In[ ]:


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


# Reading data

# Data obtained from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
dataset = pd.read_csv('data 2.csv')


# **Features** 
# 
# 1) ID number
# 
# 2) Diagnosis (M = malignant, B = benign)
# 
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# 	a) radius (mean of distances from center to points on the perimeter)
# 	b) texture (standard deviation of gray-scale values)
# 	c) perimeter
# 	d) area
# 	e) smoothness (local variation in radius lengths)
# 	f) compactness (perimeter^2 / area - 1.0)
# 	g) concavity (severity of concave portions of the contour)
# 	h) concave points (number of concave portions of the contour)
# 	i) symmetry 
# 	j) fractal dimension ("coastline approximation" - 1)
# 
# 
# The mean, standard error, and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features.

# In[ ]:


dataset.head()


# In[ ]:


print("Cancer data set dimensions : {}".format(dataset.shape))


# In[ ]:


dataset.groupby('diagnosis').size()


# In[ ]:


# Visualization of data

fig, axes = plt.subplots(ncols=10, nrows=3, figsize=(30,10))

for c, ax in zip(dataset.columns[2::], axes.flat):
    sns.distplot(dataset[dataset.diagnosis=='B'][c], ax=ax, color='g', label='benign')
    sns.distplot(dataset[dataset.diagnosis=='M'][c], ax=ax, color='r', label='malignant')
plt.legend()
plt.show()


# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset.drop(columns=['Unnamed: 32'], inplace=True)
dataset.diagnosis.replace({'B':0,'M':1}, inplace=True)

X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[ ]:


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# In[ ]:


# #Fitting the Logistic Regression Algorithm to the Training Set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, Y_train)
# #95.8 Acuracy

# #Fitting K-NN Algorithm
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, Y_train)
# #95.1 Acuracy

# #Fitting SVM
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, Y_train) 
# #97.2 Acuracy

# #Fitting K-SVM
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, Y_train)
# #96.5 Acuracy

# #Fitting Naive_Bayes
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, Y_train)
# #91.6 Acuracy

# #Fitting Decision Tree Algorithm
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, Y_train)
# #95.8 Acuracy


# In[ ]:


#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
#98.6 Acuracy


# In[ ]:


#predicting the Test set results
Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
# Train and Test Accuracy
print("Train Accuracy :: " + str(accuracy_score(Y_train, classifier.predict(X_train))))
print("Test Accuracy  :: " + str(accuracy_score(Y_test, Y_pred)))


# In[ ]:


import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, class_names=['benign','malignant'], feature_names=dataset.iloc[:, 2:32].columns)
predict_fn = lambda x: classifier.predict_proba(x)


# In[ ]:


for i in [30,31,32]:
    exp = explainer.explain_instance(X_test[i], predict_fn, num_features=5)
    exp.show_in_notebook()


# In[ ]:


df = pd.DataFrame(X_test,columns=dataset.iloc[:, 2:32].columns)
df['diagnosis'] = Y_test
df.loc[30:32,:]


# In[ ]:




