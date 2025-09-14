import os 
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(models)):
            model=list(models.values())[i]
            param=list(params.values())[i]

            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            # Train model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred=model.predict(X_test)

            # Get accuracy for test data
            model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)