# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for nltk, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout=300 --retries=10 -i https://pypi.org/simple

# Download NLTK data (including punkt_tab to fix LookupError)
RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet averaged_perceptron_tagger

# Copy all project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "spam_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
