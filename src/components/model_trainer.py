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

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        logging.info("Entered Model Trainer method/component.")

    def evaluate_model(self,X_train, X_test, y_train, y_test, models, param):
        try:
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = param[list(models.keys())[i]]

                logging.info("Hyperparameter tuning started.......(may take long)")
                gs = GridSearchCV(model, para, cv=3, n_jobs=4)
                gs.fit(X_train, y_train)
                
                logging.info("Best parameters found, applying them....")
                
                # Use the best estimator (already trained with best params)
                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)
                model_accuracy = accuracy_score(y_pred, y_test)
                report[list(models.keys())[i]] = model_accuracy
            return report
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model Training initiated.")
            models = {
                "LogisticRegression": LogisticRegression(),
                "SupportVector": SVC(),
                "KNN": KNeighborsClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "Xgboost": XGBClassifier(),
                "LDA": LinearDiscriminantAnalysis(),
                "NaiveBayes": GaussianNB()
            }

            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear"],
                    "max_iter": [1000]
                },
                "SupportVector": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                    "probability": [True]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },
                "GradientBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "Xgboost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },
                "LDA": {
                    "solver": ["svd", "lsqr"]
                },
                "NaiveBayes": {}
            }

            X_train = train_array[:, :-1]
            X_test = test_array[:, :-1]
            y_train = train_array[:, -1]
            y_test = test_array[:, -1]

            model_report = self.evaluate_model(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, 
                models=models, param=params)
            
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            logging.info(f"Best Model: {best_model_name} with accuracy: {best_model_score}")
            if best_model_score < 0.6:
                logging.info("No Model made it to threshold(60%)!!!")
            
            # Re-train the best model to get the tuned version
            best_model = models[best_model_name]
            para = params[best_model_name]
            gs = GridSearchCV(best_model, para, cv=3, n_jobs=4)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            logging.info("Saved Model Object.")

            predicted = best_model.predict(X_test)
            accu = accuracy_score(predicted, y_test)
            logging.info(f"Accuracy of model is {accu}")
            return accu

        except Exception as e:
            raise CustomException(e, sys)
