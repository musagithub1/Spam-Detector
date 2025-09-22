#!/usr/bin/env python3
import pandas as pd

def load_data(file_path):
    """Loads the spam.csv dataset and displays the first 5 rows."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        # Rename columns for clarity
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        # Drop unnecessary columns if they exist
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        print("Dataset loaded successfully. First 5 rows:")
        print(df.head())
        print("\nDistribution of labels:")
        print(df['label'].value_counts())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

if __name__ == "__main__":
    df = load_data('spam.csv')





import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data (only once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True) # Often needed for lemmatization




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


if __name__ == "__main__":
    df = load_data("spam.csv")
    if df is not None:
        print("\nApplying preprocessing to messages...")
        df["processed_message"] = df["message"].apply(preprocess_text)
        print("Preprocessing complete. First 5 processed messages:")
        print(df[["message", "processed_message"]].head())





from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(df):
    """Trains and evaluates a classification model."""
    X = df["processed_message"]
    y = df["label"]

    # Convert labels to numerical (0 for ham, 1 for spam)
    y = y.map({"ham": 0, "spam": 1})

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"\nDataset split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features to avoid overfitting and reduce dimensionality
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("Text converted to numerical features using TF-IDF.")

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

        print(f"{name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name.replace(" ", "_")}_confusion_matrix.png')
        plt.close()

    return models, tfidf_vectorizer, results

def test_model_with_samples(models, tfidf_vectorizer, sample_messages):
    """Tests the trained models with custom sample messages."""
    print("\nTesting models with custom sample messages:")
    for msg in sample_messages:
        processed_msg = preprocess_text(msg)
        msg_tfidf = tfidf_vectorizer.transform([processed_msg])
        print(f"\nOriginal: \'{msg}\'")
        print(f"Processed: \'{processed_msg}\'")
        for name, model in models.items():
            prediction = model.predict(msg_tfidf)[0]
            label = "Spam" if prediction == 1 else "Ham"
            print(f"  {name} Prediction: {label}")


if __name__ == "__main__":
    df = load_data("spam.csv")
    if df is not None:
        # Plot label distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='label', data=df, palette='viridis')
        plt.title("Distribution of Spam vs. Ham Messages")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.savefig("label_distribution.png")
        plt.close()
        print("Label distribution plot saved as label_distribution.png")

        df["processed_message"] = df["message"].apply(preprocess_text)
        print("Preprocessing complete. First 5 processed messages:")
        print(df[["message", "processed_message"]].head())

        trained_models, vectorizer, evaluation_results = train_and_evaluate_model(df)

        sample_messages = [
            "Congratulations, you won a free gift! Click here to claim.",
            "Are we meeting tomorrow at 10 AM?",
            "WINNER! You have been selected for a \$1,000,000 prize! Text CLAIM to 12345.",
            "Hey, just checking in. How are you?"
        ]
        test_model_with_samples(trained_models, vectorizer, sample_messages)


