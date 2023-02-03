# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:09:36 2019

@author: Faculty
"""

# =============================================================================
# Logistic Regression
# =============================================================================


# =============================================================================
# Business Case -- 
#The dataset comes from the UCI Machine Learning repository, and it is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. 
#The classification goal is to predict whether the client will subscribe (1/0) to a term deposit (variable y).
# =============================================================================


# =============================================================================
# Setting the Environment
# =============================================================================
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# =============================================================================
# # Importing the dataset
# =============================================================================

os.chdir('C:\Documents\IVY NOTES\PYTHON\Logistic Regression\CASE 1')
os.getcwd()
data = pd.read_csv('Banking.csv', header=0)
#data = data.dropna()
print(data.shape)
print(list(data.columns))



# =============================================================================
# # Exploratory Data Analysis
# =============================================================================


# =============================================================================
# 1. Predict variable (desired target)
# y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)
# =============================================================================

#Barplot for the dependent variable
sns.countplot(x='y',data=data, palette='hls')
plt.show()


#Check the missing values
data.isnull().sum()


#Customer job distribution
sns.countplot(y="job", data=data)
plt.show()


#Customer marital status distribution
sns.countplot(x="marital", data=data)
plt.show()


#Barplot for credit in default
sns.countplot(x="default", data=data)
plt.show()

#Barplot for housing loan


sns.countplot(x="housing",data=data)
plt.show()

#Barplot for personal loan
sns.countplot(x="loan", data=data)
plt.show()



#Barplot for previous marketing loan outcome
sns.countplot(x="poutcome", data=data)
plt.show()



# =============================================================================
# Our prediction will be based on the customer’s job, marital status, whether he(she) has credit in default, 
#whether he(she) has a housing loan, whether he(she) has a personal loan, and the outcome of the previous marketing campaigns. 
#So, we will drop the variables that we do not need.
# =============================================================================


#Dropping the redant columns
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

#Creating Dummy Variables
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

#Drop the unknown columns
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
data2.columns

#Check the independence between the independent variables
sns.heatmap(data2.corr())
plt.show()


# =============================================================================
# Split the data into training and test sets
# =============================================================================
X = data2.iloc[:,1:]#Feature Class
y = data2.iloc[:,0]# Dependent Class
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape
X_test.shape


# =============================================================================
# Fitting the Logistic Model
# =============================================================================
classifier = LogisticRegression(random_state=0)#Logistic Regression Classifier
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# =============================================================================
# Fitting the Logit Model
# =============================================================================
y_train=pd.DataFrame(y_train)

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit(method='bfgs')#Broyden (b) - Fletcher(f) - Goldfarb (g) - Shanno (S) , bfgs - Optimization methods help to eliminate the problem of singular matrix
print(result.summary2())
#If the customer is retired then the log odds of opting for term deposit increases by 0.7403

#Pseudo R-Square ideally should lie between 0.2 to 0.4
#AIC - Akai Information Criteria
#BIC - Bayesian Information Criteria ; Lower AIC and BIC more better is the model, informaton lost 
#Log - Likelihood Statistic - when Model is built all Independent Variables
#Log Likleihood Null - when Model is built only intercept has not Independent Variables
#LLR p-value: Reject the Ho: Log-Likelihood= LL-Null


print(np.exp(result.params))
#job_retired  has an ODDS Ratio of 2.096577, which indicates that if the customer is retired it increases the odds of taking up the term deposit by 2.096 times.

# =============================================================================
# Evaluating the Logistic Model
# =============================================================================


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Precision () is defined as the number of true positives () over the number of true positives plus the number of false positives ().
#Recall () is defined as the number of true positives () over the number of true positives plus the number of false negatives ().
# =============================================================================
# Interpretation:Interpretation: Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 90% of the customer’s preferred term deposits that were promoted.
# =============================================================================
