import pandas as pd
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



#statistics libraries
import statsmodels.api as sm
import scipy



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


#from pmdarima import ARIMA, auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scipy.stats import boxcox
from scipy.special import inv_boxcox



from fbprophet import Prophet

from itertools import combinations
from statsmodels.tsa.seasonal import STL


def import_data(path):
    df = (pd.read_csv(path,parse_dates=True, index_col='Date')
          .iloc[:,1:]
          .sort_index())
    df2 = df.copy()


    return df2

def clean_data(df):
    df.loc[:,'Sales'] = (df['Sales'].str.replace(',','')
                     .str.replace('$','')
                     .str.strip()
                     .astype(float))

    df.loc[:,'Fuel'] = (df['Fuel']
                    .astype('category'))
    
    return df

def fuel_data(df):
    pellet    =    df[df.Fuel=='Pellets']['Sales'].sort_index()
    pellet.name = "Pellet"
    briquette =    df[df.Fuel=='Briquettes']['Sales'].sort_index()
    briquette.name = "Briquettes"
    firewood  =    df[df.Fuel=='Firewood']['Sales'].sort_index()
    firewood.name = "Firewood"
    
    return pellet, briquette,firewood 



### Briquettes Forecast Model ###
def briquette_forecast(series_with_dates, forecast_horizon):
    
    #ETS
    series_with_dates.index.freq='MS'
    
    train = series_with_dates.values
    train_x, lam  = boxcox(series_with_dates) 
    
  
    comb_fc = combshd(train, horizon = forecast_horizon, seasonality = 12, init = 'concentrated' )
    

    
    # SARIMA
    sarima =(SARIMAX(endog=train_x,                 
             order=(1,1,1),
             seasonal_order=(1,0,1,12),
             trend='c',
             enforce_invertibility=False)).fit()
    start = len(train_x)
    end = len(train_x) + forecast_horizon -1
    
    sarima_fc = inv_boxcox(sarima.predict(start, end, dynamic=False),lam)
    
    briquette_fc = pd.Series((comb_fc + sarima_fc)/2)
    
    
    
    return briquette_fc.values

### Firewood Forecast Model ###


def firewood_forecast(series_with_dates, forecast_horizon):
    
    series_with_dates.index.freq='MS'
    #forecast_dates = series_with_dates.index.max() + 1
    train = series_with_dates.values
    train_x,lam = boxcox (train)
    hw1 = (ExponentialSmoothing(train,
                           trend='add',
                           seasonal='mul',
                           seasonal_periods=12, damped=True)).fit(use_boxcox=False)

    hw1_fc = hw1.forecast(forecast_horizon)
    
        

    
    
    train_prophet_f = pd.DataFrame({'ds':pd.date_range('1-1-2012', freq='MS',periods=len(series_with_dates)), 'y':series_with_dates.values} )
    

    prophet1=Prophet(weekly_seasonality=False,
                       yearly_seasonality=True,
                       daily_seasonality=False, 
                       n_changepoints=0, 
                       seasonality_mode="additive",
                       uncertainty_samples=0).fit(train_prophet_f) 

    fb1_df=prophet1.make_future_dataframe(forecast_horizon, freq='MS')

    prophet_fc=prophet1.predict(fb1_df)[["ds","yhat"]].tail(forecast_horizon)['yhat'].values
    
    firewood_forecast = (prophet_fc + hw1_fc )/2 #pd.DataFrame({"Sales":(hw1_fc+prophet_fc)/2},index=forecast_dates)

    
    return firewood_forecast

#### Final Pellet Model 

def pellet_forecast(train,forecast_horizon):
    
    train_x,lam = boxcox (train)
    ses=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=None,
                                           initialization_method= 'estimated', 
                                           damped_trend=False).fit()
    
    fc1 = inv_boxcox(ses.forecast(forecast_horizon),lam)
    
    holt=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=12,
                                           initialization_method= 'estimated', 
                                           damped_trend=False).fit()
    
    fc2 = inv_boxcox(holt.forecast(forecast_horizon),lam)
    
    damp=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=12,
                                           initialization_method= 'estimated', 
                                           damped_trend=True).fit()
    
    fc3 = inv_boxcox(damp.forecast(forecast_horizon),lam)
    
    fc = (fc1+fc2+fc3)/3
    return fc



