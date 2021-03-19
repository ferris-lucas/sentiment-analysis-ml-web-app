import re
import ast
import pandas as pd
from flask import Flask, render_template, request
import xgboost as xgb

app = Flask(__name__)

loaded_model = xgb.XGBClassifier(objective='reg:logistic')
loaded_model.load_model('trained_model.model')

with open("vocabulary.txt", "r") as data:
    vocabulary = ast.literal_eval(data.read())


@app.route('/')
def index():
    return render_template('index.html', prediction_text="Enter review and press predict.")


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
        msg = "According to our model, this was a positive review."
    else:
        msg = "According to our model, this was a negative review."

    return render_template("index.html", prediction_text=msg, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
