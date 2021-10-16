# Importing Libraries
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle


# Defining functions
def clean_message(msg):
    msg = msg.lower()

    stopword_list = set(stopwords.words('english'))
    msg = [word for word in msg.split() if word not in stopword_list]

    lemmatizer = WordNetLemmatizer()
    msg = [lemmatizer.lemmatize(word) for word in msg]
    msg = " ".join(msg)

    return msg


def update_model(msg, label, correct):
    # Load Data
    data = pd.read_csv('data.csv')

    # Assign the true label
    if correct:
        pass
    elif label == 'spam' and correct is False:
        label = 'ham'
    else:
        label = 'spam'

    # Cleaning message and appending to DataFrame
    cleaned_message = clean_message(msg)
    new_data = pd.DataFrame({'label': [label], 'text': [cleaned_message]})
    data = pd.concat([data, new_data], ignore_index=True)

    # Splitting data for modelling
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.1, random_state=42)

    # Training Models
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    model = svm.SVC(C=10)
    model.fit(X_train, y_train)

    # Save model and update data
    with open('vectorizer.pkl', 'wb') as filename:
        pickle.dump(vectorizer, filename)
    with open('model.pkl', 'wb') as filename:
        pickle.dump(model, filename)

    # Saving updated data
    data.to_csv('data.csv', index=False)