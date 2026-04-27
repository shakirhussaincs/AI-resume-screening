from sentence_transformers import SentenceTransformer, util
import os

# 1. Importing your shared preprocessing
try:
    from preprocessing import clean_resume_text
except ImportError:
    def clean_resume_text(text): return str(text).lower().strip()

# Load the BERT model globally (so it doesn't reload on every function call)
# 'all-MiniLM-L6-v2' is the best balance of speed and accuracy
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(resume_text, job_description_text):
    """
    Calculates semantic similarity using BERT.
    No longer requires tfidf_vectorizer.pkl or feature_selector.pkl.
    """
    # 1. Clean the text
    resume_clean = clean_resume_text(resume_text)
    job_clean = clean_resume_text(job_description_text)
    
    if not resume_clean or not job_clean:
        return 0.0
    try:
        resume_embedding = model.encode(resume_clean, convert_to_tensor=True)
        job_embedding = model.encode(job_clean, convert_to_tensor=True)
        
        # Raw Cosine Similarity (The 0.5 you are seeing)
        raw_score = float(util.cos_sim(resume_embedding, job_embedding).item())
        
        # CALIBRATION LOGIC:
        # If score is 0.5, we want it to look like ~90%
        # If score is 0.2, it's actually quite poor.
        if raw_score > 0.4:
            # Boosts scores above 0.4 significantly
            calibrated_score = 0.5 + (raw_score * 0.5) 
        else:
            calibrated_score = raw_score * 1.2 # Slight boost for lower scores
            
        # Ensure we never exceed 100%
        final_score = min(calibrated_score, 0.9999)
        
        return round(final_score, 4)
        
    except Exception as e:
        return 0.0
def calculate_batch_similarity(resumes_dict, job_description):
    # This remains the same as your previous logic
    results = []
    for filename, text in resumes_dict.items():
        score = calculate_similarity(text, job_description)
        results.append({"filename": filename, "score": score})
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    print("Testing BERT Semantic Similarity...")
    test_resume = "Software Engineer proficient in Python and AI"
    test_job = "Python Machine Learning Developer"
    
    score = calculate_similarity(test_resume, test_job)
    print(f"✅ BERT Match Score: {score * 100:.2f}%")