from flask import Flask, jsonify, request
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            area = float(data["area"])

            lin_reg = joblib.load("./LR_ser.pkl")
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(area).tolist())

#
@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("./training_data.pkl")

            df = pd.read_json(data)

            df_training_set= df


            new_lin_reg = LinearRegression()
            new_lin_reg.fit(df_training_set['sqm_living'], df_training_set['price'])

            os.remove("./linear_regression_model.pkl")
            os.remove("./training_data.pkl")

            joblib.dump(new_lin_reg, "linear_regression_model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")

            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")



if __name__ == '__main__':
    app.run(debug=True)