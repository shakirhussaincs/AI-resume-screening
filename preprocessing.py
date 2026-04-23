import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Run these once in your terminal if not done already:
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_resume_text(text):
    # Step 1: Convert to string (safety check)
    text = str(text)

    # Step 2: Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)

    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Step 4: Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Step 5: Lowercase everything
    text = text.lower()

    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    # Step 8: Lemmatization (reduce to root word)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)