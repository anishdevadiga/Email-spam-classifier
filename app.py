import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt_tab')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)


# Load the models
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Email SMS Spam Classifier")

# Input and preprocess
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms:
        transform_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transform_sms])

        # Make prediction
        result_sms = model.predict(vector_input)

        if result_sms[0] == 1:
            st.header("🚨 Spam Message / Email")
        else:
            st.header("✅ Not a Spam Message or Email")

