# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 02:42:18 2021

Trains and Saves vectorizer and svm model for application in UI

@author: Hikikonut
"""

# Importing Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

try:
    import nltk
except:
    import pip
    pip.main(['install', 'nltk'])
    import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Initialise variables for local file paths
WORKING_DIRECTORY = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIRECTORY, 'spam_data')


def clean_message(msg):
    msg = msg.lower()
    
    stopword_list = set(stopwords.words('english'))
    msg = [word for word in msg.split() if word not in stopword_list]

    lemmatizer = WordNetLemmatizer()
    msg = [lemmatizer.lemmatize(word) for word in msg]
    msg = " ".join(msg)
    return msg


# IMPORT DATA
data1 = pd.read_csv(os.path.join(DATA_DIR, 'spam1.csv'), encoding='latin-1')
data2 = pd.read_csv(os.path.join(DATA_DIR, 'spam2.csv'), encoding='latin-1')

# DATA CLEANING
data1 = data1[['v1', 'v2']]
data1 = data1.rename(columns = {'v1': 'label', 'v2': 'text'})
data1['text'] = data1['text'].apply(clean_message)

data2 = data2[['label','email']]
data2 = data2.rename(columns = {'email': 'text'})
data2['label'].replace([0,1], ['ham','spam'], inplace=True)
data2.dropna(inplace=True)

data = pd.concat([data1, data2], axis = 0)
data.to_csv('data.csv', index=False)
# print(data.shape)

# Splitting data for modelling 
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], 
                                                    test_size = 0.25, random_state = 42)

# Training Models
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

model = svm.SVC(C=10)
model.fit(X_train, y_train)

# Saving Models
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Testing Model
X_test = vectorizer.transform(X_test)
y_pred = model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))

# Plotting Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), 
                              display_labels=model.classes_)
disp.plot()
plt.show()
