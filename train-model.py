import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP ---
export_dir = 'MODEL'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# --- 2. LOAD PRE-CLEANED DATA ---
# We load the version you already cleaned and saved to save time
print("Loading pre-cleaned dataset...")
df = pd.read_json('Dataset/cleaned_resumes.jsonl', lines=True)

# Important: Fill any NaNs that might have appeared during saving/loading
df['cleaned_text'] = df['cleaned_text'].fillna('')

# --- 3. ENCODING (The Numeric Mapping) ---
le = LabelEncoder()
y = le.fit_transform(df['Category']) 
joblib.dump(le, os.path.join(export_dir, 'label_encoder.pkl'))
print(f"✅ Label Encoder saved ({len(le.classes_)} categories).")

# --- 4. VECTORIZATION ---
tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    max_df=0.8,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=15000 
)
X = tfidf.fit_transform(df['cleaned_text'])
joblib.dump(tfidf, os.path.join(export_dir, 'tfidf_vectorizer.pkl'))
print(f"✅ TF-IDF Vectorizer saved (Features: {X.shape[1]}).")

# --- 5. FEATURE SELECTION (The Noise Filter) ---
# We use the NUMERIC 'y' here to avoid the error you found earlier
selector = SelectKBest(chi2, k=10000)
X_selected = selector.fit_transform(X, y) 
joblib.dump(selector, os.path.join(export_dir, 'feature_selector.pkl'))
print(f"✅ Feature Selector saved (Reduced to 10,000 features).")

# --- 6. FINAL MODEL TRAINING ---
print("Training the Final Production Random Forest...")
final_rf = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    n_jobs=-1, 
    random_state=42
)
final_rf.fit(X_selected, y)
joblib.dump(final_rf, os.path.join(export_dir, 'resume_rf_model.pkl'))

print(f"\n🚀 ALL COMPONENTS MATHEMATICALLY ALIGNED!")
print(f"Location: /{export_dir}")