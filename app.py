import streamlit as st
import pickle
import re
import numpy as np
import nltk

# Download required NLTK data (for Streamlit Cloud)
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Article Text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        confidence = round(max(probability) * 100, 2)

        # Show prediction
        if prediction == 1:
            st.success(f"This News is REAL ✅")
        else:
            st.error(f"This News is FAKE ❌")

        st.info(f"Confidence Score: {confidence}%")

        # ----- Explanation Part -----
        st.subheader("🔍 Top Influencing Words")

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        # Get non-zero features in this input
        input_array = vectorized.toarray()[0]
        nonzero_indices = np.where(input_array > 0)[0]

        word_contributions = []

        for idx in nonzero_indices:
            contribution = input_array[idx] * coefficients[idx]
            word_contributions.append((feature_names[idx], contribution))

        # Sort by absolute impact
        word_contributions = sorted(word_contributions, key=lambda x: abs(x[1]), reverse=True)

        top_words = word_contributions[:5]

        for word, score in top_words:
            st.write(f"• {word} (impact score: {round(score, 4)})")

        st.markdown("---")
        st.subheader("📖 Explanation")

        if prediction == 1:
            st.write("The article contains words and patterns commonly found in reliable news reporting.")
        else:
            st.write("The article contains words and patterns commonly associated with misleading or fake news.")