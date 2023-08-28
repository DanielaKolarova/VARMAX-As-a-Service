from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
from itertools import product
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

# declare constants 
MODEL_FILE_NAME = 'web-app/savedmodels/././varmax_model.pkl'

def test_model_prediction():
    data = load_data()
  
    train, test = train_test_split(data, test_size=0.2)
    exog_cols = data.columns.drop(['year', 'quarter', 'realgdp', 'realcons'])
    print(exog_cols)
    exog_test = test[exog_cols]
    
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    print(model.summary())
    predict = model.forecast(steps=len(test), exog=exog_test)
    predict = predict.reset_index(drop=True)
    predict.rename(columns = {'realgdp':'realgdp_predicted', 'realcons':'realcons_predicted'}, inplace = True)
    
    endog_test = test[['realgdp', 'realcons']].reset_index(drop=True)
    comparison_df = pd.concat([endog_test, predict], axis=1)
    
    print(comparison_df)


def load_data():
    macro_data = sm.datasets.macrodata.load_pandas()
    
    return macro_data.data

if __name__=='__main__':
    test_model_prediction()