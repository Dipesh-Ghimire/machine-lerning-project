import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransConfig
from src.components.model_trainer import (ModelTrainerConfig,ModelTrainer)

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    # construct file paths dynamically in your code : /artifacts/train.csv
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(
                "/home/dipesh/Desktop/End-To-End ML Project/notebook/data/stud.csv")
            # Instead of local system, data can also be fetched from cloud, MongoDB, APIs in any format

            logging.info("Read the dataset as Dataframe")

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("Train-test-split is initiated")
            train_set, test_set = train_test_split(
                df, test_size=.2, random_state=42)
            # saving train_set.csv and test_set.csv
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)
            logging.info("Ingestion of Data is Completed.")
            logging.info(f"train=={self.ingestion_config.train_data_path},test =={self.ingestion_config.test_data_path},")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr)) #0.8785718776814511