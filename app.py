import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from datetime import datetime

courses = pd.read_csv('./365_database/365_course_info.csv').values.tolist()

countries =[['1', 'Usa'], ['2', 'India'], ['3', 'Canada'], ['4', 'Australia'], ['5', 'United Kingdom'], ['6', 'Germany'], ['7', 'France'], ['8', 'Brazil'], ['9', 'China'], ['10', 'Japan'], ['11', 'Russia'], ['12', 'Italy'], ['13', 'Spain'], ['14', 'Mexico'], ['15', 'South Korea'], ['16', 'Netherlands'], ['17', 'Switzerland'], ['18', 'Sweden'], ['19', 'Belgium'], ['20', 'Singapore'], ['21', 'Turkey'], ['22', 'Austria'], ['23', 'Norway'], ['24', 'Denmark'], ['25', 'Ireland'], ['26', 'Poland'], ['27', 'Saudi Arabia'], ['28', 'Ukraine'], ['29', 'Czech Republic'], ['30', 'Portugal']]

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', courses=courses, countries=countries)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]

    course = int(request.form['course_id'])
    minutes = int(request.form['minutes_watched'])
    country = int(request.form['country_id'])
    course_rating = int(request.form['course_rating'])

    period = pd.to_datetime(request.form['date_watched']) - pd.to_datetime(request.form['date_registered'])
    period_bf_rating = pd.to_datetime(request.form['date_rated']) - pd.to_datetime(request.form['date_watched'])
    
    period = np.abs(int(period.days))
    period_bf_rating = np.abs(int(period_bf_rating.days))

    final_features = [course, minutes, country, course_rating, period, period_bf_rating]
    prediction = model.predict([final_features])
    proba = model.predict_proba([final_features])
    output = prediction[0]
    yes = round(proba[0][1]*100, 2) 
    no = round(proba[0][0]*100, 2)

    return render_template('result.html', predictiont=output, yes=yes, no=no)

if __name__ == "__main__":
    app.run(debug=True)