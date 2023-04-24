# Principle: There is no GOAT Algorithm because no one know which algorithm will perform best on your dataset
# Solution is try each and every algorithm and see how they perform on your dataset
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    # Output of data_trainsformation becomes input of model_trainer::train_array,test_array

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and test Input Data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], test_array[:, :-1], test_array[:,-1]
            )
            # Dictionary of Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            #To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            #To get best model name from dict :: using List Comprehension
            best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
               ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
               raise CustomException("No Best Model Found best_model_score < 0.6")

            logging.info("Best Model Found")

            # Remember the Pre-processing Pickel File can be loaded in case of New Datasets

            save_object(
               file_path= self.model_trainer_config.trained_model_file_path,
               obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score( y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
