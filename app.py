from flask import Flask, request,render_template
import pandas as pd
import pickle

app = Flask(__name__)
xgb_model = pickle.load(open('XGBoost_Classifier.pkl','rb'))

@app.route("/")
def get_model():
    # return str(xgb_model)
    return render_template("form.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = pd.read_csv('./data/data.csv')
    data.drop(["Unnamed: 32","id"], axis=1, inplace=True)

    X = data.drop('diagnosis', axis=1)

    cols = X.columns

    series = {}

    for col in cols:
        val = request.form[col]
        series[col] = [float(val)]

    vec = pd.DataFrame(series)
    names = xgb_model.get_booster().feature_names

    norm = (vec - X.mean())/(X.max()-X.min())

    pred = xgb_model.predict(vec[names].iloc[[-1]])
    if pred == 1:
        return "Prediction : Benign Tumour Found"
    else:
        return "Prediction: Malignant Tumour Found"

if __name__=="__main__":
    app.run(debug=True)


