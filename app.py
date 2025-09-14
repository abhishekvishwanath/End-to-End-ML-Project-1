from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),  
            parental_level_of_education=request.form.get('parental_level_of_education'),  
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),  
            reading_score=int(request.form.get('reading_score')),  
            writing_score=int(request.form.get('writing_score'))  
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)
