# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:08:23 2023

@author: hamed
"""

# Import Laibraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix,precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



path = 'C:\\Users\\hamed\\Desktop\\classifcation\\drug.txt'
data = pd.read_csv(path, header = None, names=['Age','Sex','BP','Cholesterol',
                                               'Na_to_K','Drug'])
#print(data.info())
#print(data.isnull().sum())

drug_counts = data['Drug'].value_counts()
plt.figure(figsize=(8,6))
plt.bar(drug_counts.index, drug_counts.values)
plt.xlabel('Drug Type')
plt.ylabel('Count')
plt.title('Distribution of Drug Types')
plt.show()


plt.figure(figsize=(8, 6))
plt.pie(drug_counts.values, labels=drug_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Drug Types')
plt.axis('equal')  
plt.show()


label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['BP'] = label_encoder.fit_transform(data['BP'])
data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])
data['Drug'] = label_encoder.fit_transform(data['Drug'])


X = data.drop('Drug', axis =1)
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=.2, random_state=42)



knn_classifier = KNeighborsClassifier()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()
nb_classifier = GaussianNB()
svc_classifier = SVC()
lg_classifier = LogisticRegression()

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


