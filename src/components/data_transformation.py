import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransConfig()

    def get_data_transformer_object(self):
        # This function is responsible for Data Transformation : OHE, Standard Scaler
        try:
            num_col = ['writing_score', 'reading_score']
            cat_col = ["gender", "race_ethnicity",
                       "parental_level_of_education", "lunch", "test_preparation_course"]
            # Creating a Pipleline which
            # 1.Handle missing values using SimpleImputer (median or mode)
            # 2. Stardard Scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Columns: {num_col}")
            logging.info(f"Categorical Columns: {cat_col}")
            # Combination of num_pipeline and cat_pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col)
                ]
            )
            # (name-of-pipeline, pipeline, columns to transform with above pipeline)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data Completed")
            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformer_object()

            num_col = ['writing_score', 'reading_score']
            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying pre-processing object on training dataframe and testing dataframe")
            #Most Crucial Part of Code where preprocessing is actaully done
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing Object")

            # saving pickel file
            save_object(
                file_path=self.data_trans_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_trans_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
