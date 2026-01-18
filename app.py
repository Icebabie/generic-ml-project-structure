from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    prediction = "not_set"  # Change from None to a sentinel value
    if request.method == "GET":
        return render_template('index.html', prediction=prediction)
    else:
        data = CustomData(
            Pregnancies = int(request.form.get('Pregnancies')),
            Glucose = float(request.form.get('Glucose')),
            BloodPressure = float(request.form.get('BloodPressure')),
            SkinThickness = float(request.form.get('SkinThickness')),
            Insulin = float(request.form.get('Insulin')),
            BMI = float(request.form.get('BMI')),
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
            Age = int(request.form.get('Age'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.Predict(pred_df)

        return render_template(
            "index.html",
            prediction=int(results[0])
        )


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)