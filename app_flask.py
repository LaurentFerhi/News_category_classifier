from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle as pk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__, static_url_path='/static')

### Load pickle files
# Load vectorizer
with open("vectorizer.pkl", 'rb') as fid:
    vectorizer_trained = pk.load(fid)

# Load categories
with open("categ.pkl", 'rb') as fid:
    categ = pk.load(fid)
    
# Load categories
with open("classifier.pkl", 'rb') as fid:
    trained_clf = pk.load(fid)

# Parsing text + cleaning + stemm
def parse_out_text(all_text):
    # clean punctuation, make lower case and remove stopwords
    text_string = all_text.translate(str.maketrans("", "", string.punctuation)).split(" ")
    text_string = [word.lower() for word in text_string if word.lower() not in stopwords.words('english')]
    # Stemm text
    stemmer = SnowballStemmer("english")
    stemmed = [stemmer.stem(word) for  word in text_string]
    words = " ".join(stemmed) 
    return words

def make_prediction(text_to_predict):
	prepared_text = parse_out_text(text_to_predict)
	feat_array = vectorizer_trained.transform([prepared_text]).toarray()
	result_int = trained_clf.predict(feat_array)[0]
	return categ[result_int]

### Flask application
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    text = [x for x in request.form.values()][0]
    prediction = "Predicted category: "+make_prediction(text)
    return render_template('home.html',pred='{}'.format(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
