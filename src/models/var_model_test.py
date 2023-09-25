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
from storage.minio_api import load_model

import warnings
warnings.filterwarnings('ignore')


def test_model_prediction():
    
    data = load_data()
    endog = data[['realgdp', 'realcons']]
  
    endog_train, endog_test = train_test_split(endog, test_size=0.1, shuffle=False)
    print(endog_test)
    
    model = load_model('var', '0_0_1')
    print(model.summary())
    
    predict = model.forecast(steps=len(endog_test))
    predict = predict.reset_index(drop=True)
    predict.rename(columns = {'realgdp':'realgdp_predicted', 'realcons':'realcons_predicted'}, inplace = True)
    
    endog_test = endog_test.reset_index(drop=True)
    comparison_df = pd.concat([endog_test, predict], axis=1)
    
    print(comparison_df)

def load_data():
    macro_data = sm.datasets.macrodata.load_pandas()
    
    return macro_data.data

if __name__=='__main__':
    test_model_prediction()