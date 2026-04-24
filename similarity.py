from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def _basic_clean(text):
    """
    Independent cleaning logic for similarity calculation.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def calculate_similarity(resume_text, job_description_text):
    """
    Calculates TF-IDF cosine similarity between resume and job description.
    """
    if not resume_text or not job_description_text:
        return 0.0
    
    resume_clean = _basic_clean(resume_text)
    job_clean = _basic_clean(job_description_text)
    
    if not resume_clean or not job_clean:
        return 0.0
        
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 4)
    except Exception:
        return 0.0

def calculate_batch_similarity(resumes_dict, job_description):
    """
    Independent batch processor for a dictionary of resumes.
    """
    results = []
    for filename, text in resumes_dict.items():
        score = calculate_similarity(text, job_description)
        results.append({"filename": filename, "score": score})
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    print("Testing Independent Similarity Engine...")
    test_resumes = {"resume1.pdf": "Python Developer", "resume2.pdf": "Java Developer"}
    job = "Python Developer"
    scores = calculate_batch_similarity(test_resumes, job)
    for res in scores:
        print(f"{res['filename']}: {res['score'] * 100}%")
