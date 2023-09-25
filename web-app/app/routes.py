# coding=utf-8

from flask import render_template, flash, redirect, session, url_for, request, g, Markup
from app import app
import boto3, botocore

import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

# declare constants
BUCKET_NAME = 'models'

with open('/var/www/varmax-as-a-service/savedmodels/varmax_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/varmax_predict')
def predict_with_varmax():
    """ Endpoint returning a macroeconomic time series prediction based on a statistical model called VARMAX
    ---
    parameters:
      - name: realinv
        in: query
        type: number
        required: true
      - name: realgovt
        in: query
        type: number
        required: true
      - name: realdpi
        in: query
        type: number
        required: true
      - name: cpi
        in: query
        type: number
        required: true
      - name: m1
        in: query
        type: number
        required: true
      - name: tbilrate
        in: query
        type: number
        required: true
      - name: unemp
        in: query
        type: number
        required: true
      - name: pop
        in: query
        type: number
        required: true
      - name: infl
        in: query
        type: number
        required: true
      - name: realint
        in: query
        type: number
        required: true
    responses:
      200:
        description: "Predicted data"
      
    """
    
    realinv = request.args.get("realinv")
    realgovt = request.args.get("realgovt")
    realdpi = request.args.get("realdpi")
    cpi = request.args.get("cpi")   
    m1 = request.args.get("m1")
    tbilrate = request.args.get("tbilrate")
    unemp = request.args.get("unemp")
    pop = request.args.get("pop")
    infl = request.args.get("infl")
    realint = request.args.get("realint")
    
    exog_vars = np.array([[realinv, realgovt, realdpi, cpi, m1, tbilrate, unemp, pop, infl, realint]]);
    
    prediction = model.predict(steps=len(exog_vars), exog = exog_vars)
    prediction = prediction.reset_index(drop=True)
    
    print(prediction)
    
    return str(prediction)

@app.route('/varmax_predict_file', methods=["POST"])
def predict_with_varmax_file():
    """File endpoint returning a macroeconomic time series prediction based on a statistical model called VARMAX
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    exog_vars = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(steps=len(exog_vars), exog = exog_vars)
    
    return str(list(prediction))
  

@app.route('/var_predict/<version>')
def predict_with_var(version):
    """ Endpoint returning a macroeconomic time series prediction based on a statistical model called VAR
    ---
    parameters:
      - name: future_periods
        in: query
        type: number
        required: true
      - name: version
        in: path
        type: string
        required: true
    responses:
      200:
        description: "Predicted data"
      
    """
    future_periods = request.args.get('future_periods', type=int)
    
    var_model = _load_model('var', version)
    
    prediction = var_model.predict(start = var_model.nobs, end=var_model.nobs + future_periods)
    prediction = prediction.reset_index(drop=True)
    
    return str(prediction)
  
def _load_model(name, version):
    """
        Loads the model stored in a minio bucket suffixed by the specified version:
        
        name - model name
        version - model version 
    """
    s3client = boto3.client('s3', 
        endpoint_url='http://host.docker.internal:9100/',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )

    object_name = _build_object_name(name, version)

    response = s3client.get_object(Bucket=BUCKET_NAME, Key=object_name)
    body = response['Body'].read()
    model = pickle.loads(body)   
        
    return model

def _build_object_name(name, version):
    return ''.join([name, '/', version, '/', 'model.pkl'])  

