from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle

from storage.minio_api import store_model

import warnings
warnings.filterwarnings('ignore')


def run_model_pipeline():
    """
        Runs the algorithm for model preparation:
        
        - Load data
        - Divide into train and test dataset (test dataset is not used for now)
        - Prepare endogenous variables
        - Find optimal model parameters (p) model
        - Instantiate model with the optimal parameters identified
        - Serialize the instantiated model to a pickle file and save to an S3 object storage (MinioIO)
    """
    
    data = load_data()
    endog = data[['realgdp', 'realcons']]

    endog_train, endog_test = train_test_split(endog, test_size=0.1, shuffle=False)
    
    optimized_model_parameters = optimize_var(endog_train)
    optimized_model = instatiate_model(endog_train, optimized_model_parameters)
    serialize_model(optimized_model)
    
    print('VAR model trained and serialized!')

def load_data():
    """
        Loads preprocessed macroeconomic data from https://www.statsmodels.org/dev/datasets/generated/macrodata.html
    """
    macro_data = sm.datasets.macrodata.load_pandas()
    
    return macro_data.data

def optimize_var(endog):
    """
        Returns a dataframe with parameters and corresponding MSE
        
        endog - observed time series
    """
    
    results = []
    
    for p in range(15):
        try:
            model = VARMAX(endog, order=(p, 0)).fit(dips=False)
        except:
            print("An error occured for parameter: ", p)
            continue
           
        mse = model.mse
        results.append([p, mse])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['p', 'mse']
    
    result_df = result_df.sort_values(by='mse', ascending=True).reset_index(drop=True)
    print(result_df)
    
    return [result_df['p'].iloc[0], 0]

def instatiate_model(train_endog, optimized_model_parameters):
    """
        Constructs an instanve of the VARMAX model based on the parameters passed
        
        train_endog - the observed variable (training data)
        optimized_model_parameters - optimal p and q expected to be 0 (p, q) pair
    """
    
    best_model_VAR = VARMAX(endog=train_endog, order=optimized_model_parameters)
    optimized_model = best_model_VAR.fit(disp=False)
        
    return optimized_model

def serialize_model(optimized_model):
    """
        Stores a model as a pickle file into an S3 storage
        
        optimized_model - the model for serialization
    """
    store_model('var', '0_0_1', optimized_model)


if __name__=='__main__':
    run_model_pipeline()