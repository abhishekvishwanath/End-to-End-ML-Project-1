import os 
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_data,test_data):
        try:
            X_train=train_data[:,:-1]
            y_train=train_data[:,-1]   
            X_test=test_data[:,:-1]
            y_test=test_data[:,-1]
            logging.info("Splitting of data into dependent and independent features is completed")

            models={
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(), 
                "XGBRegressor": XGBRegressor(),  
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }     

            params={
                "Linear Regression": {},
                "Ridge": {'alpha':[0.1,1.0,10.0,100.0]},
                "K-Neighbors Regressor": {'n_neighbors':[3,5,7,9]},
                "Decision Tree": {'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                                  'max_depth':[None,2,3,5,10],
                                  'min_samples_split':[2,5,10]},
                "Random Forest Regressor": {'n_estimators':[50,100,200],
                                            'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                                            'max_depth':[None,2,3,5,10],
                                            'min_samples_split':[2,5,10]},
                "XGBRegressor": {'learning_rate':[0.01,0.1,0.2,0.3],
                                 'n_estimators':[50,100,200],
                                 'max_depth':[3,4,5,6]},
                "CatBoosting Regressor": {'depth':[3,4,5,6],
                                           'learning_rate':[0.01,0.1,0.2,0.3],
                                           'iterations':[30,50,100]},
                "AdaBoost Regressor": {'learning_rate':[0.01,0.1,0.2,0.3],
                                       'n_estimators':[50,100,200]}
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params=params)
            logging.info(f"Model Report : {model_report}")

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            logging.info(f"Best Model Found : {best_model_name} with score {best_model_score}")

            if best_model_score<0.6:
                raise CustomException("No best model found")    
            
            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model
            )
            logging.info("Best model is saved")
            return best_model_name,best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)

