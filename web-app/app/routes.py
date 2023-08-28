# coding=utf-8

from flask import render_template, flash, redirect, session, url_for, request, g, Markup
from app import app

import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

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
    """ Endpoint returning a macroeconomic time series prediction based on a statistical model
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
    
    return str(prediction)

@app.route('/varmax_predict_file', methods=["POST"])
def predict_with_varmax_file():
    """File endpoint returning a macroeconomic time series prediction based on a statistical model
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


