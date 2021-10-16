from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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


def predict_spam(msg):
    # Cleaning Message
    cleaned_message = clean_message(msg)

    # Load Models
    with open('vectorizer.pkl', 'rb') as filename:
        vectorizer = pickle.load(filename)
    with open('model.pkl', 'rb') as filename:
        model = pickle.load(filename)

    # Predict outcome
    cleaned_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(cleaned_message)

    return prediction[0]

