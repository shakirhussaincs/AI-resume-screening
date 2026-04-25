from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
# Importing from your teammates' files as requested
try:
    from preprocessing import clean_resume_text
except ImportError:
    def clean_resume_text(text): return str(text).lower()

try:
    # Importing the class reference from the training script
    from train_model import TfidfVectorizer
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_similarity(resume_text, job_description_text):
    """
    Calculates similarity using imports from preprocessing and train_model.
    """
    # 1. Clean using teammate's logic
    resume_clean = clean_resume_text(resume_text)
    job_clean = clean_resume_text(job_description_text)
    
    if not resume_clean or not job_clean:
        return 0.0
    
    # 2. Check for pre-trained components in MODEL folder
    vectorizer_path = 'MODEL/tfidf_vectorizer.pkl'
    selector_path = 'MODEL/feature_selector.pkl'
    
    try:
        if os.path.exists(vectorizer_path):
            tfidf = joblib.load(vectorizer_path)
            v_res = tfidf.transform([resume_clean])
            v_job = tfidf.transform([job_clean])
            
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                v_res = selector.transform(v_res)
                v_job = selector.transform(v_job)
                
            similarity = cosine_similarity(v_res, v_job)[0][0]
        else:
            # Fallback if no model is found
            tfidf = TfidfVectorizer()
            vectors = tfidf.fit_transform([resume_clean, job_clean])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
        return round(float(similarity), 4)
    except Exception:
        return 0.0

def calculate_batch_similarity(resumes_dict, job_description):
    results = []
    for filename, text in resumes_dict.items():
        score = calculate_similarity(text, job_description)
        results.append({"filename": filename, "score": score})
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    print("Testing Similarity with External Imports...")
    test_resume = "Python Developer with machine learning experience"
    test_job = "Python AI Developer"
    score = calculate_similarity(test_resume, test_job)
    print(f"Match Score: {score * 100}%")
