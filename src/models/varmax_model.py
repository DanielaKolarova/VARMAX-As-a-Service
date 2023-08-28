from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
from itertools import product
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

# declare constants 
MODEL_FILE_NAME = 'web-app/savedmodels/././varmax_model.pkl'

def run_model_pipeline():
    """
        Runs the algorithm for model preparation:
        
        - Load data
        - Divide into train and test dataset (test dataset is not used for now)
        - Prepare exogenous variables
        - Find optimal parameters (p, q) model
        - Instatiate model with the optimal parameters identified
        - Serialize the instatiated model to a pickle file 
    """
    
    data = load_data()

    train, test = train_test_split(data, test_size=0.2)
    
    endog_train = train[['realgdp', 'realcons']]
    exog_cols = data.columns.drop(['year', 'quarter', 'realgdp', 'realcons'])
    exog_train = train[exog_cols]
    
    optimized_model_parameters = optimize_varmax(endog_train, exog_train)
    optimized_model = instatiate_model(endog_train, exog_train, optimized_model_parameters)
    serialize_model(optimized_model)
    
    print('VARMAX model trained and serialized!')

def load_data():
    """
        Loads preprocessed macroeconomic data from https://www.statsmodels.org/dev/datasets/generated/macrodata.html
    """
    macro_data = sm.datasets.macrodata.load_pandas()
    
    return macro_data.data

def optimize_varmax(train_endog, train_exog):
    """
        Returns a dataframe with (p,q) and MSE
        
        train_endog - the observed variable (training data)
        train_exog - the exogenous variables (training data)
    """
    
    p = range(0, 4, 1)
    q = range(0, 4, 1)

    parameters = product(p, q)
    parameters_list = list(parameters)
    
    results = []
    
    for param in parameters_list:
        try:
            print("Iteration started with (p,q): ", param)
            model = VARMAX(train_endog, train_exog, order=param).fit(disp=False)
        except BaseException:
            print("An error occured for parameter: ", param)
            continue
    
        mse = model.mse
        results.append([param, mse])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'mse']
    
    result_df = result_df.sort_values(by='mse', ascending=True).reset_index(drop=True)
    print(result_df)
    
    return result_df['(p,q)'].iloc[0]

def instatiate_model(train_endog, train_exog, optimized_model_parameters):
    """
        Constructs an instanve of the VARMAX model based on the parameters passed
        
        train_endog - the observed variable (training data)
        train_exog - the exogenous variables (training data)
        optimized_model_parameters - optimal (p, q) pair
    """
    
    best_model_VARMAX = VARMAX(train_endog, train_exog, order=optimized_model_parameters)
    optimized_model = best_model_VARMAX.fit(disp=False)
    
    return optimized_model

def serialize_model(optimized_model):
    """
        Stores a model as a pickle file
        
        optimized_model - the model fr serialization
    """
    pickle.dump(optimized_model, open(MODEL_FILE_NAME,'wb'))


if __name__=='__main__':
    run_model_pipeline()