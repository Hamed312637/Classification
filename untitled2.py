# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:08:39 2023

@author: hamed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix,precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

path  = 'C:\\Users\\hamed\\Desktop\\classifcation\\train.csv'


train_data = pd.read_csv(path, header=None, names=['PassengerId','Survived','Pclass','Name',
                                                   'Sex','Age','SibSp','Parch','Ticket','Fare',
                                                   'Cabin','Embarked'])





print(train_data.isnull().sum())
train_data['Age'].fillna(train_data['Age'].median(skipna = True),inplace = True)
train_data['Embarked'].fillna(train_data['Embarked'].value_counts().idxmax(),inplace = True)
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)
training.drop('Cabin',axis =1 ,inplace = True)
final_train = training
print(final_train.head())
X= final_train.drop('Survived', axis=1)
y = final_train['Survived']





X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=.2, random_state=42)
knn_classifier = KNeighborsClassifier()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()
nb_classifier = GaussianNB()
svc_classifier = SVC(kernel='poly')
lg_classifier = LogisticRegression(penalty='l2', solver='saga',)

model =[knn_classifier, dt_classifier, rf_classifier, nb_classifier, svc_classifier, 
        lg_classifier]
model_names = ['KNN', 'Decision Tree', 'Random Forest', 'Gaussian Naive Bayes',
                'Support Vector Classifier','Logistic Regression']


accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

for m in model:
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, average='weighted')
    precision = precision_score(y_test,y_test, average = 'weighted',zero_division = 1)
    recall = recall_score(y_test, y_predict, average='weighted')
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
x = np.arange(len(model_names))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x, accuracy_scores, width, label='Accuracy')
plt.bar(x + width, f1_scores, width, label='F1 Score')
plt.bar(x + (2 * width), precision_scores, width, label='Precision')
plt.bar(x + (3 * width), recall_scores, width, label='Recall')

plt.xlabel('Classifier')
plt.ylabel('Score')
plt.title('Performance Evaluation of Classifiers')
plt.xticks(x + width * 1.5, model_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


