# It will have all common functions that we use during entire project
import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
     Serialize and save a Python object to disk.

     Parameters:
         file_path (str): The path where the serialized object will be saved.
         obj (object): The Python object to be serialized and saved.

     Raises:
         CustomException: If any errors occur during the execution of the function.

     Returns:
         None
     """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
     Evaluates the performance of multiple machine learning models on a given dataset using cross-validation.

     Parameters:
         X_train (array-like): The training dataset.
         y_train (array-like): The target labels for the training dataset.
         X_test (array-like): The test dataset.
         y_test (array-like): The target labels for the test dataset.
         models (dict): A dictionary containing the machine learning models to be evaluated.
         params (dict): A dictionary containing the hyperparameters to be tuned for each model.

     Returns:
         dict: A dictionary containing the R-squared scores for each model on the test data.
     """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train Model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
     Load a serialized Python object from disk.

     Parameters:
         file_path (str): The path where the serialized object is saved.

     Raises:
         CustomException: If any errors occur during the execution of the function.

     Returns:
         object: The Python object that was serialized and saved.
     """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)