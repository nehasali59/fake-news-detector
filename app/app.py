from flask import Flask, request, render_template
import pickle
import os
import numpy as np
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_model = pickle.load(open(os.path.join(BASE_DIR, 'model/saved_model/lr_model.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'model/saved_model/tfidf_vectorizer.pkl'), 'rb'))

# LIME explainer
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

def predict_proba(texts):
    tfidf_features = tfidf.transform(texts)
    return lr_model.predict_proba(tfidf_features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form['article']

    # Prediction
    tfidf_input = tfidf.transform([article])
    prediction = lr_model.predict(tfidf_input)[0]
    probability = lr_model.predict_proba(tfidf_input)[0]

    result = "Real News" if prediction == 1 else "Fake News"
    confidence = round(probability[prediction] * 100, 2)

    # LIME explanation
    explanation = explainer.explain_instance(
        article,
        predict_proba,
        num_features=10
    )

    lime_words = explanation.as_list()
    positive_words = [w for w, s in lime_words if s > 0]
    negative_words = [w for w, s in lime_words if s < 0]

    return render_template('index.html',
        result=result,
        confidence=confidence,
        positive_words=positive_words,
        negative_words=negative_words,
        article=article
    )

if __name__ == '__main__':
    app.run(debug=True)