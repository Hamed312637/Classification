# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:31:13 2023

@author: hamed
"""


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



path = 'C:\\Users\\hamed\\Desktop\\classifcation\\data.txt'
df = pd.read_csv(path, header = None, names=["id","diagnosis","radius_mean","texture_mean",
                                               "perimeter_mean","area_mean","smoothness_mean",
                                               "compactness_mean","concavity_mean",
                                               "concave points_mean","symmetry_mean",
                                               "fractal_dimension_mean","radius_se",
                                               "texture_se","perimeter_se","area_se",
                                               "smoothness_se","compactness_se","concavity_se",
                                               "concave points_se","symmetry_se",
                                               "fractal_dimension_se","radius_worst",
                                               "texture_worst","perimeter_worst","area_worst",
                                               "smoothness_worst","compactness_worst",
                                               "concavity_worst","concave points_worst",
                                               "symmetry_worst","fractal_dimension_worst",])

#print(data.info())
print(df.isnull().sum())

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])


X= df.drop('diagnosis',axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=.2, random_state=42)


knn_classifier = KNeighborsClassifier()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()
nb_classifier = GaussianNB()
svc_classifier = SVC()
lg_classifier = LogisticRegression()

model = [knn_classifier, dt_classifier, rf_classifier, nb_classifier, svc_classifier, 
        lg_classifier]

model_names = ['KNN', 'Decision Tree', 'Random Forest', 'Gaussian Naive Bayes',
                'Support Vector Classifier','Logistic Regression']

accuracy_scores =[]
f1_scores = []
precision_scores = []
recall_scores =[]

for i in model :
    i.fit(X_train, y_train)
    y_predict = i.predict(X_test)
    accuracy = accuracy_score(y_test,y_predict)
    f1 = f1_score(y_test, y_predict,average='weighted')
    precision = precision_score(y_test, y_predict,average='weighted',zero_division=1)
    recall = recall_score(y_test, y_predict,average='weighted')
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
x = np.arange(len(model_names))
width =.2
plt.figure(figsize=(12,6))
plt.bar(x, accuracy_scores,width, label = 'Accuracy')
plt.bar(x+ width, f1_scores,width , label = 'Scores')
plt.bar(x+(2*width), precision_scores,width , label = 'Precision')    
plt.bar(x+(3*width),recall_scores, width, label = 'Recall' )
plt.xlabel('Classifier')
plt.ylabel('Score')
plt.title('Performance Evaluation of Classifiers')
plt.xticks(x + width * 1.5, model_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()    
print(accuracy_scores)    

