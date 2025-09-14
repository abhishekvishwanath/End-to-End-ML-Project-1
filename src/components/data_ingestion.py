import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join('artifacts','data.csv')
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")
        try:
            df=pd.read_csv('notebooks/data/StudentsPerformance.csv')
            logging.info("Dataset has been read as dataframe")

            # Rename columns to use consistent underscore naming
            df.rename(columns={
                "math score": "math_score",
                "reading score": "reading_score", 
                "writing score": "writing_score",
                "parental level of education": "parental_level_of_education",
                "test preparation course": "test_preparation_course",
                "race/ethnicity": "race_ethnicity"
            }, inplace=True)
            logging.info("Column names standardized to underscore format")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data has been stored in data.csv")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Train and Test data has been stored in train.csv and test.csv")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion_obj=DataIngestion()
    train_data,test_data=data_ingestion_obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer=ModelTrainer()
    model,score=model_trainer.initiate_model_trainer(train_arr,test_arr)

    print(model,score)
