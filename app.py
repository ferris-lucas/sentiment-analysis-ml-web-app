import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import joblib
import ast
import pandas as pd
from flask import Flask, render_template, request
import json
import pickle
import xgboost as xgb


app = Flask(__name__)

loaded_model = xgb.XGBClassifier(objective='reg:logistic')
loaded_model.load_model('trained_model.model')

with open("vocabulary.txt", "r") as data:
    vocabulary = ast.literal_eval(data.read())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', ])
def make_prediction():

    if request.method == 'POST':
        user_input = request.form['user_input']
        input_review = re.sub('(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])', "", user_input)
        input_review = re.sub('(<br\s*/><br\s*/>)|(\-)|(\/)', " ", input_review)
        bow = [0] * len(vocabulary)
        for word in input_review.split():
            if word in vocabulary:
                bow[vocabulary[word]] += 1
        review_bow = bow
        review_bow = pd.DataFrame(review_bow).T

        [prediction] = loaded_model.predict(review_bow)

    if prediction == 1:
        msg = "Positive review."
    else:
        msg = "Negative review."

    return render_template("index.html",   prediction_text=msg, user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True)