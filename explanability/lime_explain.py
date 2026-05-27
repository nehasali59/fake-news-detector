import pickle
import numpy as np
from lime.lime_text import LimeTextExplainer
import os

# Load saved model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_model = pickle.load(open(os.path.join(BASE_DIR, 'model/saved_model/lr_model.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'model/saved_model/tfidf_vectorizer.pkl'), 'rb'))
# lr_model = pickle.load(open('../model/saved_model/lr_model.pkl', 'rb'))
# tfidf = pickle.load(open('../model/saved_model/tfidf_vectorizer.pkl', 'rb'))

# Create predict function for LIME
def predict_proba(texts):
    tfidf_features = tfidf.transform(texts)
    return lr_model.predict_proba(tfidf_features)

# Create LIME explainer
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

# Test article
test_article = """
Scientists have discovered that drinking coffee every day 
can cure all types of cancer instantly without any side effects.
Doctors are shocked by this miracle discovery.
"""

# Generate explanation
explanation = explainer.explain_instance(
    test_article,
    predict_proba,
    num_features=10
)

# Print results
print("Prediction:", "Real" if lr_model.predict(tfidf.transform([test_article]))[0] == 1 else "Fake")
print("\nTop words influencing prediction:")
for word, weight in explanation.as_list():
    print(f"  {word}: {weight:.4f}")