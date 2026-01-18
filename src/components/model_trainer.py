import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



# All classical ML algorithms for small dataset classification problem with binary target and numerical features.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        logging.info("Entered Model Trainer method/component.")

    def evaluate_model(self,X_train, X_test, y_train, y_test, models):
        try:
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train)
                y_pred= model.predict(X_test)
                model_accuracy = accuracy_score(y_pred,y_test)
                report[list(models.keys())[i]] = model_accuracy
            return report
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model Training initiated.")
            models = {
                "LogisticRegression": LogisticRegression(),
                "SupportVector":SVC(),
                "KNN": KNeighborsClassifier(),
                "RandomForest":RandomForestClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "Xgboost" : XGBClassifier(),
                "LDA": LinearDiscriminantAnalysis(),
                "NaiveBayes": GaussianNB()
            }

            X_train=train_array[:,:-1]
            X_test=test_array[:,:-1]
            y_train=train_array[:,-1]
            y_test=test_array[:,-1]

            model_report:dict = self.evaluate_model(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,models=models)
            
            best_model_score=max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info("Best Model found...")
            if best_model_score<0.6:
                logging.info("No Model made it to threshhold(60%)!!!")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
                )
            logging.info("Saved Model Object.")
    
            predicted = best_model.predict(X_test)
            accu = accuracy_score(predicted,y_test)
            return accu


        except Exception as e:
            raise CustomException(e,sys)
        