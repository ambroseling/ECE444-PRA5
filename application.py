from flask import Flask, request, g, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

# @application.route("/")
# def index():
#     return "Your Flask App Works! V1.0"

def load_model():
    g.model = None
    with open("basic_classifier.pkl", "rb") as f:
        g.model = pickle.load(f)
        print("Model loaded")
    g.vectorizer = None 
    with open("count_vectorizer.pkl", "rb") as f:
        g.vectorizer = pickle.load(f)
        print("Vectorizer loaded")

@application.before_request
def before_request():
    if not hasattr(g, 'model') or not hasattr(g, 'vectorizer'):
        load_model()
    assert g.model is not None and g.vectorizer is not None, "Model or vectorizer not loaded"

@application.route("/")
def index():
    return render_template("predict.html")

@application.route("/predict", methods=["POST"])
def predict():
    data = request.json 
    if data is None:
        data = {}
        data["text"]= request.form.get("text")
    vectorized_input = g.vectorizer.transform([data["text"]])
    prediction = g.model.predict(vectorized_input)[0]
    return json.dumps({"prediction": prediction})

if __name__ == "__main__":
    application.run(port=5000, debug=True)