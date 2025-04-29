import streamlit as st
import speech_recognition as sr
import joblib
import re, string

# Load model and vectorizer
model = joblib.load("savedlogistic_regression_model.pkl")
vectorizer = joblib.load("savedtfidf_vectorizer.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = ' '.join(text.split())
    return text

# Audio file transcription
def recognize_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return None

# Streamlit UI
st.title("Speech Fraud Classifier")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    text = recognize_audio(uploaded_file)
    if text:
        st.write("**Recognized Text:**", text)
        cleaned = preprocess_text(text)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]
        st.success("Prediction: Fraudulent" if pred == 1 else "Prediction: Genuine")
    else:
        st.error("Could not transcribe audio.")
