# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:39:13 2023

@author: hamed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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



def label_encoder_fit(x):
    unique = np.unique(x)
    labels = {value:i for i , value in enumerate(unique)}
    return labels


def label_encoder_transform(x,labels):
    return [labels[value]for value in x]



labels = label_encoder_fit(df['diagnosis'])
df['diagnosis']  = label_encoder_transform(df['diagnosis'], labels)
#print(df['diagnosis'])
def feature_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled

def data_split(data,split_ratio):
    now_rows = df.shape[0]
    test_rows = int(now_rows* split_ratio)
    test_indicaties = random.sample(range(now_rows),test_rows)
    test_set = df.iloc[test_indicaties]
    train_set = df.drop(test_indicaties)
    return train_set, test_set

train, test = data_split(df, 0.3)
X_train = train.drop(columns=['diagnosis']).values
y_train = train["diagnosis"].values
X_test = test.drop(columns=['diagnosis']).values
y_test = test["diagnosis"].values

X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

def predict(X,W,b):
    return W.dot(X)+b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(W,b):
   m,n = X_train.shape
   cost = 0
   for i in range(m):
       fx = sigmoid(predict(X_train[i], W, b))
       cost += y_train[i]*np.log(fx)+(1-y_train[i])*np.log(1-fx)
   cost = -cost/m
   return cost

def gradient_step(W, b):
    m,n = X_train.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        fx = sigmoid(predict(X_train[i], W, b))
        for j in range(n):
            dj_dw[j] += (fx - y_train[i]) * X_train[i][j]
        dj_db += fx - y_train[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return [dj_dw, dj_db]

def gradient_decent(alpha, iterations, init_W, init_b, record_interval):
    local_W = init_W
    local_b = init_b
    cost_history = [cost(local_W, local_b)]
    for i in range(iterations):
        new_W, new_b = gradient_step(local_W, local_b)
        local_W = local_W - (alpha*new_W)
        local_b = local_b - (alpha*new_b)
        if i%record_interval==0:
            local_cost = cost(local_W, local_b)
            print(f"Iteration {i}: Cost = {local_cost}")
            cost_history.append(local_cost)
    return [local_W, local_b, cost_history]


m, n = X_train.shape
W = np.zeros((n,))
b = 0
alpha = 0.001  
iterations = 20000 
variable_frequency = 1000 


W, b, cost_history = gradient_decent(alpha, iterations, W, b,variable_frequency)


print('Final weights:')
print(W)
print('Final bias:')
print(b)

plt.plot(range(0, iterations + 1, variable_frequency), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Change over Iterations')
plt.show()


linear_combination = np.dot(X_test, W) + b
probabilities = sigmoid(linear_combination)
predictions = np.where(probabilities >= 0.4, 1, 0)
accuracy = np.mean(predictions == y_test) * 100
print("Accuracy:", accuracy, "%")