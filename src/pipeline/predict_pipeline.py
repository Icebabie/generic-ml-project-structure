import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
    
class CustomData:
    def __init__(self,
        Pregnancies:int,
        Glucose:int,
        BloodPressure:int,
        SkinThickness:int,
        Insulin:float,
        BMI:float,
        DiabetesPedigreeFunction:int,
        Age:int):

        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness=SkinThickness
        self.Insulin=Insulin
        self.BMI=BMI
        self.DiabetesPedigreeFunction=DiabetesPedigreeFunction
        self.Age=Age

    def get_data_as_data_frame(self):
        try:
            custom_input_data_dict={
                "Pregnancies":[self.Pregnancies],
                "Glucose":[self.Glucose],
                "BloodPressure":[self.BloodPressure],
                "SkinThickness":[self.SkinThickness],
                "Insulin":[self.Insulin],
                "BMI":[self.BMI],
                "DiabetesPedigreeFunction":[self.DiabetesPedigreeFunction],
                "Age":[self.Age]
            }

            return pd.DataFrame(custom_input_data_dict)
        except Exception as e:
            raise CustomException(e,sys)  