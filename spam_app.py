#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Download necessary NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

download_nltk_data()

def preprocess_text(text):
    """Applies a series of preprocessing steps to the input text."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@st.cache_data
def load_data():
    """Loads the spam.csv dataset."""
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        # Rename columns for clarity
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        # Drop unnecessary columns if they exist
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        return df
    except FileNotFoundError:
        st.error("Error: The file spam.csv was not found.")
        return None

@st.cache_resource
def train_models():
    """Trains and returns the models and vectorizer."""
    df = load_data()
    if df is None:
        return None, None
    
    # Preprocess the data
    df["processed_message"] = df["message"].apply(preprocess_text)
    
    X = df["processed_message"]
    y = df["label"].map({"ham": 0, "spam": 1})
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    # Train models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_tfidf, y)
    
    return models, tfidf_vectorizer

def main():
    st.title("📧 Spam Message Detection")
    st.write("Enter a message below to check if it's spam or ham (legitimate).")
    
    # Load models
    models, vectorizer = train_models()
    
    if models is None or vectorizer is None:
        st.error("Failed to load models. Please ensure spam.csv is available.")
        return
    
    # Text input
    user_input = st.text_area("Enter your message:", height=100, placeholder="Type your message here...")
    
    if st.button("Predict"):
        if user_input.strip():
            # Preprocess the input
            processed_input = preprocess_text(user_input)
            
            # Transform using TF-IDF
            input_tfidf = vectorizer.transform([processed_input])
            
            # Make predictions
            st.subheader("Predictions:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nb_pred = models["Multinomial Naive Bayes"].predict(input_tfidf)[0]
                nb_label = "🚨 Spam" if nb_pred == 1 else "✅ Ham"
                st.metric("Naive Bayes", nb_label)
            
            with col2:
                lr_pred = models["Logistic Regression"].predict(input_tfidf)[0]
                lr_label = "🚨 Spam" if lr_pred == 1 else "✅ Ham"
                st.metric("Logistic Regression", lr_label)
            
            with col3:
                svm_pred = models["Support Vector Machine"].predict(input_tfidf)[0]
                svm_label = "🚨 Spam" if svm_pred == 1 else "✅ Ham"
                st.metric("Support Vector Machine", svm_label)
            
            # Show processed text
            st.subheader("Processed Text:")
            st.code(processed_input)
            
            # Consensus prediction
            predictions = [nb_pred, lr_pred, svm_pred]
            spam_count = sum(predictions)
            
            if spam_count >= 2:
                st.error("🚨 **Consensus: This message is likely SPAM**")
            else:
                st.success("✅ **Consensus: This message is likely HAM (legitimate)**")
        else:
            st.warning("Please enter a message to analyze.")
    
    # Show some example messages
    st.subheader("Try these examples:")
    
    examples = [
        "Congratulations, you won a free gift! Click here to claim.",
        "Are we meeting tomorrow at 10 AM?",
        "WINNER! You have been selected for a $1,000,000 prize! Text CLAIM to 12345.",
        "Hey, just checking in. How are you?"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}: {example[:50]}...", key=f"example_{i}"):
            st.text_area("Enter your message:", value=example, height=100, key=f"example_text_{i}")

if __name__ == "__main__":
    main()

