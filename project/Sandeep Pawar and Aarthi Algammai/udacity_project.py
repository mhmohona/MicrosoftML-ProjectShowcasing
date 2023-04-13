import numpy as np
import itertools

def MAPE(y_true, y_pred): 
    """
    %Error compares true value with predicted value. Lower the better. Use this along with rmse(). If the series has 
    outliers, compare/select model using MAPE instead of rmse()
    
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def residcheck(residuals, lags):
    """
    Function to check if the residuals are white noise. Ideally the residuals should be uncorrelated, zero mean, 
    constant variance and normally distributed. First two are must, while last two are good to have. 
    If the first two are not met, we have not fully captured the information from the data for prediction. 
    Consider different model and/or add exogenous variable. 
    
    If Ljung Box test shows p> 0.05, the residuals as a group are white noise. Some lags might still be significant. 
    
    Lags should be min(2*seasonal_period, T/5)
    
    plots from: https://tomaugspurger.github.io/modern-7-timeseries.html
    
    """
    resid_mean = np.mean(residuals)
    lj_p_val = np.mean(ljung(x=residuals, lags=lags)[1])
    norm_p_val =  jb(residuals)[1]
    adfuller_p = adfuller(residuals)[1]
    
    
    
    fig = plt.figure(figsize=(10,8))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2);
    acf_ax = plt.subplot2grid(layout, (1, 0));
    kde_ax = plt.subplot2grid(layout, (1, 1));

    residuals.plot(ax=ts_ax)
    plot_acf(residuals, lags=lags, ax=acf_ax);
    sns.kdeplot(residuals);
    #[ax.set_xlim(1.5) for ax in [acf_ax, kde_ax]]
    sns.despine()
    plt.tight_layout();
    
    print("** Mean of the residuals: ", np.around(resid_mean,2))
    
    print("\n** Ljung Box Test, p-value:", np.around(lj_p_val,3), "(>0.05, Uncorrelated)" if (lj_p_val > 0.05) else "(<0.05, Correlated)")
    
    print("\n** Jarque Bera Normality Test, p_value:", np.around(norm_p_val,3), "(>0.05, Normal)" if (norm_p_val>0.05) else "(<0.05, Not-normal)")
    
    print("\n** AD Fuller, p_value:", np.around(adfuller_p,3), "(>0.05, Non-stationary)" if (adfuller_p > 0.05) else "(<0.05, Stationary)")
    
    
    
    return ts_ax, acf_ax, kde_ax

    
def accuracy(y1,y2):
    
    accuracy_df=pd.DataFrame()
    
    rms_error = np.round(rmse(y1, y2),1)
    
    map_error = np.round(np.mean(np.abs((np.array(y1) - np.array(y2)) / np.array(y1))) * 100,1)
           
    accuracy_df=accuracy_df.append({"RMSE":rms_error, "%MAPE": map_error}, ignore_index=True)
    
    return accuracy_df

def plot_pgram(series,diff_order):
    """
    This function plots thd Power Spectral Density of a de-trended series. 
    PSD should also be calculated for a de-trended time series. Enter the order of differencing needed
    Output is a plot with PSD on Y and Time period on X axis
    
    Series: Pandas time series or np array
    differencing_order: int. Typically 1
    
    """
    from scipy import signal    
    de_trended = series.diff(diff_order).dropna()
    f, fx = signal.periodogram(de_trended)
    freq=f.reshape(len(f),1) #reshape the array to a column
    psd = fx.reshape(len(f),1)
#     plt.figure(figsize=(5, 4)
    plt.plot(1/freq, psd  )
    plt.title("Periodogram")
    plt.xlabel("Time Period")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    


    
def hw_cv(series, seasonal_periods, initial_train_window, test_window):
    from statsmodels.tools.eval_measures import rmse
    import warnings
    warnings.filterwarnings("ignore")
    """
     Author: Sandeep Pawar
     Date: 9/17/2020
     Ver: 2.0
     Returns Rolling and Expanding cross-validation scores (avg rmse), along with model paramters
     for Triple Exponential Smoothing method. Expanding expands the training set each time by adding one observation,
     while rolling slides the training and test by one observation each time.
     Output shows parameters used and Rolling & Expanding cv scores. Output is in below order:
          1. Trend 2. Seasonal 3. Damped 4. use_boxcox 5. Rolling cv 6. Expanding cv
     Requirements: Pandas, Numpy, Statsmodels, itertools, rmse
     series: Pandas Series
             Time series
     seasonal_periods: int
             No of seasonal periods in a full cycle (e.g. 4 in quarter, 12 in monthly, 52 in weekly data)
     initial_train_window: int
             Minimum training set length. Recommended to use minimum 2 * seasonal_periods
     test_window: int
             Test set length. Recommended to use equal to forecast horizon
     e.g. hw_cv(ts["Sales"], 4, 12, 6 )
          Output: add add False False    R: 41.3   ,E: 39.9
     Note: This function can take anywhere from 5-15 min to run full output
    """
    def expanding_tscv(series,trend,seasonal,seasonal_periods,damped,boxcox,initial_train_window, test_window):
        i =  0
        x = initial_train_window
        t = test_window
        errors_roll=[]
        while (i+x+t) <len(series):
            train_ts=series[:(i+x)].values
            test_ts= series[(i+x):(i+x+t)].values
            model_roll = ExponentialSmoothing(train_ts,
                                         trend=trend,
                                         seasonal=seasonal,
                                         seasonal_periods=seasonal_periods,
                                         damped=damped).fit(use_boxcox=boxcox)
            fcast = model_roll.forecast(t)
            error_roll = rmse(test_ts, fcast)
            errors_roll.append(error_roll)
            i += 1
        return np.mean(errors_roll).round(1)

    def rolling_tscv(series,trend,seasonal,seasonal_periods,damped,boxcox,initial_train_window, test_window):
        i =  0
        x = initial_train_window
        t = test_window
        errors_roll=[]
        while (i+x+t) <len(series):
            train_ts=series[(i):(i+x)].values
            test_ts= series[(i+x):(i+x+t)].values
            model_roll = ExponentialSmoothing(train_ts,
                                         trend=trend,
                                         seasonal=seasonal,
                                         seasonal_periods=seasonal_periods,
                                         damped=damped).fit(use_boxcox=boxcox)
            fcast = model_roll.forecast(t)
            error_roll = rmse(test_ts, fcast)
            errors_roll.append(error_roll)
            i += 1
        return np.mean(errors_roll).round(1)

    trend      = ['add','mul']
    seasonal   = ['add','mul']
    damped     = [False, True]
    use_boxcox = [False, True, 'log']
    params = itertools.product(trend,seasonal,damped,use_boxcox)
    # series, seasonal_periods, initial_train_window, test_window
    for trend,seasonal,damped,use_boxcox in params:
        #r=rolling_tscv(series, trend, seasonal, 4, damped, use_boxcox, 12,4)
        e=expanding_tscv(series, trend, seasonal, seasonal_periods, damped, use_boxcox, initial_train_window,test_window)
        result = print(trend, seasonal, damped, use_boxcox,"    ,Expanding RMSE :", e)
    return result





# Bagged ETS
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ets

def cbb(array):
    i=0
    l = array.shape[0]
    z = []
    while i < l:
        s = np.tile(array, (l+1))
        y = s[i:i+l]
        z.append(y)
        i += 1
    return z

def baggedets(series, h = 6, seasonal_periods=4, s_window=7, initialization = 'heuristic', damped=True, averaging ='mean'):
    
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    import warnings
    warnings.filterwarnings("ignore")

    #Transform to BoxCox
    #bcox, lam = boxcox(series)
    bcox = np.log(series)

    #Seasonal Decomposition using STL, seasonal shoul dbe >= 7 and odd 
    stl = (STL(bcox,
               seasonal = s_window,
               period = seasonal_periods).fit())


    bag = []


    for i in range(stl.resid.shape[0]):
        recon = stl.trend + stl.seasonal + cbb(stl.resid)[i]
        bag.append(recon)

    fc_list = []

    for i in range(stl.resid.shape[0]):

        model = (ets(bag[i],
                     trend=True,
                     damped_trend=damped,
                     seasonal=seasonal_periods,
                     initialization_method=initialization ).fit())

        fc_list.append(model.forecast(h))


        if averaging == 'mean':
            forecast = np.exp((np.mean(fc_list, axis=0)))
        else:
            forecast = np.exp((np.median(fc_list, axis=0)))


    return forecast


def pysnaive(train_series,seasonal_periods,forecast_horizon):
    '''
    Python implementation of Seasonal Naive Forecast. 
    This should work similar to https://otexts.com/fpp2/simple-methods.html
    Returns two arrays
     > fitted: Values fitted to the training dataset
     > fcast: seasonal naive forecast
    
    Author: Sandeep Pawar
    
    Date: Apr 9, 2020
    
    Ver: 1.0
    
    train_series: Pandas Series
        Training Series to be used for forecasting. This should be a valid Pandas Series. 
        Length of the Training set should be greater than or equal to number of seasonal periods
        
    Seasonal_periods: int
        No of seasonal periods
        Yearly=1
        Quarterly=4
        Monthly=12
        Weekly=52
        

    Forecast_horizon: int
        Number of values to forecast into the future
    
    e.g. 
    fitted_values = pysnaive(train,12,12)[0]
    fcast_values = pysnaive(train,12,12)[1]
    '''
    
    if len(train_series)>= seasonal_periods: #checking if there are enough observations in the training data
        
        last_season=train_series.iloc[-seasonal_periods:]
        
        reps=np.int(np.ceil(forecast_horizon/seasonal_periods))
        
        fcarray=np.tile(last_season,reps)
        
        fcast=pd.Series(fcarray[:forecast_horizon])
        
        fitted = train_series.shift(seasonal_periods)
        
    else:
        fcast=print("Length of the trainining set must be greater than number of seasonal periods") 
    
    return fitted, fcast

def combinations(results_df):

    '''Author: Sandeep Pawar
    Date: 8/30/2020
    version: 1.2'''


    models = ['SES', 'Holt', 'Damp', 'HW1', 'Naive', 'ETS1', 'ETS_BoxCox','SARIMA', 'Prophet','BaggedETS']
    b_list = []
    for i in np.arange(1,len(models)+1):
        lst = list(itertools.combinations(models,i))
        b_list.append(lst)

    m = [item for t in b_list for item in t] 
    rmse_list = pd.DataFrame(columns=['Models','RMSE'])
    for i in np.arange(1,len(m)):
             median = results_df[list(m[i])].median(axis=1).values
             e = rmse(results_df.Test,median).round(1)
             rmse_list=rmse_list.append({'Models':str(list(m[i])),'RMSE':e},ignore_index=True)
    return rmse_list


def combshd(train,horizon,seasonality, init):

    '''Author: Sandeep Pawar
    Date: 8/30/2020
    version: 1.1'''
    
    train_x,lam = boxcox (train)
    ses=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=None,
                                           initialization_method= init, 
                                           damped_trend=False).fit()

    fc1 = inv_boxcox(ses.forecast(horizon),lam)

    holt=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=seasonality,
                                           initialization_method= init, 
                                           damped_trend=False).fit()

    fc2 = inv_boxcox(holt.forecast(horizon),lam)

    damp=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=seasonality,
                                           initialization_method= init, 
                                           damped_trend=True).fit()

    fc3 = inv_boxcox(damp.forecast(horizon),lam)

    return (fc1+fc2+fc3)/3


def ts(series, fc_horizon, seasonality, hw_trend, hw_seasonal, hw_damped, hw_bcox, sarima_O1, sarima_O2, exog_series):
    '''
    
    Author: Sandeep Pawar
    Date: 8/27/2020
    version: 1
    
    Run hw_grid first identify hw params
    Run pmdarima to find SARIMA orders
        
    
    '''
    df = pd.DataFrame()
    
    length = len(series)
    test = series.tail(fc_horizon).values
    train = series.head(length-fc_horizon).values
    
    df['Test']= test
    
    exog_test = exog_series.tail(fc_horizon).values
    exog_train = exog_series.head(length-fc_horizon).values
    
    train_x,lam = boxcox(train)
    
    # 1. Single Exponential Smoothing
    ses=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=None,
                                           initialization_method= 'estimated', 
                                           damped_trend=False).fit()
    
    ses_fc = inv_boxcox(ses.forecast(fc_horizon),lam)
    
    df['SES']= ses_fc
    
    #2. Holt Additive
    holt=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=seasonality,
                                           initialization_method= 'estimated', 
                                           damped_trend=False).fit()
    
    holt_fc = inv_boxcox(holt.forecast(fc_horizon),lam)
    
    df['Holt']= holt_fc
    
    #3. Damped Trend
    damp=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           seasonal=seasonality,
                                           initialization_method= 'estimated', 
                                           damped_trend=True).fit()
    
    damp_fc = inv_boxcox(damp.forecast(fc_horizon),lam)
    
    df['Damp']= damp_fc
    
    #4. Holt's Trend
    hw1 = (ExponentialSmoothing(train,
                           trend=hw_trend,
                           seasonal=hw_seasonal,
                           seasonal_periods=seasonality, damped=hw_damped)).fit(use_boxcox=hw_bcox)

    hw1_fc = hw1.forecast(fc_horizon)
    
    df['HW1']= hw1_fc
    
    #5. Naive
    naive, snaive_fc = pysnaive(pd.Series(train),seasonality,fc_horizon)
    
    df['Naive']= snaive_fc.values
    
    #6. ETS 
    ets1=sm.tsa.statespace.ExponentialSmoothing(train,
                                           trend=True, 
                                           initialization_method= 'concentrated', 
                                           seasonal=seasonality, 
                                           damped_trend=False).fit()
    
    ets1_fc = ets1.forecast(fc_horizon)
    
    #7. ETS with BoxCox
    df['ETS1']= ets1_fc
    
    ets2=sm.tsa.statespace.ExponentialSmoothing(train_x,
                                           trend=True, 
                                           initialization_method= 'concentrated', 
                                           seasonal=seasonality, 
                                           damped_trend=True).fit()
    
    ets2_fc = inv_boxcox(ets2.forecast(fc_horizon),lam)
    
    df['ETS_BoxCox']= ets2_fc
    
    #8. SARIMA
    sarima =(SARIMAX(endog=train_x,                 
             order=(0,1,1),
             seasonal_order=(1,0,1,12),
             trend='c',
             enforce_invertibility=False)).fit()
    start = len(train)
    end = len(train) + len(test) -1
    
    sarima_fc = inv_boxcox(sarima.predict(start, end, dynamic=False),lam)
    df['SARIMA']= sarima_fc
    
    #9. SARIMAX
    sarimax =(SARIMAX(endog=train_x,                 
             exog = exog_train,
             order=sarima_O1,
             seasonal_order=sarima_O2,
             trend='c',
             enforce_invertibility=False)).fit()
    
    sarimax_fc = inv_boxcox(sarimax.predict(start, end, dynamic=False, exog=exog_test),lam)
    df['SARIMAX']= sarimax_fc
    
    #10. Prophet
    train_prophet1 = pd.DataFrame({'ds':pd.date_range('1-1-2012', freq='MS',periods=len(train)), 'y':train} )
    

    prophet1=Prophet(weekly_seasonality=False,
                       yearly_seasonality=True,
                       daily_seasonality=False, 
                       n_changepoints=10, 
                       seasonality_mode="additive",
                       uncertainty_samples=0).fit(train_prophet1) 

    fb1_df=prophet1.make_future_dataframe(fc_horizon, freq='MS')

    prophet_fc=prophet1.predict(fb1_df)[["ds","yhat"]].tail(len(test))['yhat'].values

    df['Prophet']= prophet_fc
    
    #11. CombSHD
    df['CombSHD'] = (df['SES'] + df['Holt'] + df['Damp'])/3
    
    #12. BaggedETS
    df['BaggedETS'] = (baggedets(train, 
                                 h = fc_horizon, 
                                 seasonal_periods=seasonality, 
                                 s_window=13, 
                                 initialization = 'concentrated',
                                 damped=True, averaging ='median'))
    #13. Median
    df['Median'] = df.drop(['Test','CombSHD'], axis=1).median(axis=1).values
    
    
        
    return  df 

