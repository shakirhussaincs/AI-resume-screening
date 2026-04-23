import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def has_corrupted_text(text):
    corrupted_patterns = ['iclaire', 'aclaired', 'eclaire', 'uclaire']
    return any(pattern in text.lower() for pattern in corrupted_patterns)

def clean_resume_text(text):
    """
    Clean and preprocess resume text for ML model
    
    Steps:
    1. Remove URLs, emails, special characters
    2. Convert to lowercase
    3. Remove stopwords
    4. Lemmatize words to root form
    """
    # Step 1: Convert to string (safety check)
    text = str(text)
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    
    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Step 4: Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Step 5: Lowercase
    text = text.lower()
    
    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 7: Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Step 8: Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)