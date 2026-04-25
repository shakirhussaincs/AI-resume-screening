from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
# Importing TfidfVectorizer and clean_resume_text from preprocessing as requested
from preprocessing import TfidfVectorizer, clean_resume_text

def calculate_similarity(resume_text, job_description_text):
    """
    Calculates similarity using the Vectorizer imported from preprocessing.py.
    """
    # Use the shared cleaning function
    resume_clean = clean_resume_text(resume_text)
    job_clean = clean_resume_text(job_description_text)
    
    if not resume_clean or not job_clean:
        return 0.0
    
    # Check if a pre-trained vectorizer exists in the MODEL folder
    vectorizer_path = 'MODEL/tfidf_vectorizer.pkl'
    selector_path = 'MODEL/feature_selector.pkl'
    
    try:
        if os.path.exists(vectorizer_path):
            # Use the pre-trained vectorizer from train-model.py
            tfidf = joblib.load(vectorizer_path)
            
            # Transform the texts
            vectors = tfidf.transform([resume_clean, job_clean])
            
            # If a selector was used during training, apply it here too
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                vectors = selector.transform(vectors)
            
            # Calculate similarity on the transformed (and potentially selected) vectors
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        else:
            # Fallback: Create a fresh vectorizer if no model is trained
            tfidf = TfidfVectorizer(stop_words='english')
            vectors = tfidf.fit_transform([resume_clean, job_clean])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
        return round(float(similarity), 4)
    except Exception as e:
        print(f"Similarity Error: {e}")
        return 0.0

def calculate_batch_similarity(resumes_dict, job_description):
    """
    Processes a bundle of resumes using the synced logic.
    """
    results = []
    for filename, text in resumes_dict.items():
        score = calculate_similarity(text, job_description)
        results.append({"filename": filename, "score": score})
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SYNCED SIMILARITY ENGINE TEST ")
    print("="*60)
    test_resume = "Python Developer with machine learning experience"
    test_job = "Python Developer for AI project"
    score = calculate_similarity(test_resume, test_job)
    print(f"Match Score: {score * 100}%")
    print("Logic: Imported via preprocessing.py")
    print("="*60)
