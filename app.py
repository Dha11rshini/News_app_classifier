import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean input text
def clean_input(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())  # Remove non-letters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit page config
st.set_page_config(page_title="Fake News Classifier", page_icon="📰")

# Stylish header
st.markdown("""
    <h1 style='text-align: center; color: #3366cc;'>📰 Fake vs Real News Classifier</h1>
    <p style='text-align: center; color: gray;'>Paste a news article or headline to check if it's fake or real</p>
""", unsafe_allow_html=True)

# Text input from user
input_text = st.text_area("Enter News Text 👇", height=200)

# Predict button
if st.button("Predict 🔍"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_input(input_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]  # Probability of being real

        if prediction == 1:
            st.success("✅ This news looks **REAL**!")
            st.info(f"🧠 Confidence: {proba:.2f}")
        else:
            st.error("❌ This news seems **FAKE**!")
            st.info(f"🧠 Confidence: {1 - proba:.2f}")

# Footer
st.markdown("<hr><p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
