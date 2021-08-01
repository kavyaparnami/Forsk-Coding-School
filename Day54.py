# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:13:58 2021

@author: lenovo
"""

import pandas as pd
df =  pd.read_csv('balanced_reviews.csv')
df.isnull().any(axis = 0)
df.dropna(inplace =  True)

df = df [df['overall'] != 3]

import numpy as np
df['Positivity'] = np.where(df['overall'] > 3 , 1 , 0)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, predictions)

from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, predictions)

import pickle
file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))