# Sales Forecasting Using Azure ML

**Team** : Sandeep Pawar, Aarthi Alagammai

  
  

**Introduction:**

Forecasting is one of the key business metrics essential for planning for any organization. We have created a Sales Forecasting solution using real sales data from an Irish fuel supplier. After reading the forecasting blog of one of the team members (Sandeep), the sales manager from this fuel supplier contacted him and asked if would like to help him create forecast for rest of the year. This data is from Sept 2013 to Mar 2020, total 78 months.

Link to the Sales Forecasting dashboard: [Forecast Dashboard ](https://app.powerbi.com/view?r=eyJrIjoiMTU2Y2MzMDctYzkwNC00MjA0LTlmOTktODNkZWQ4MDZjNmU1IiwidCI6IjkxMzc2MWU4LTc4NjEtNDc0ZS05ZjM4LWQyZDc1MjUwMDExZiJ9&pageName=ReportSection76d20a236ccf778e58a4)

  

!['Demo'](https://raw.githubusercontent.com/pawarbi/datasets/master/powerbi_gif.gif)

  

**Data Description:**

  

- Date : Sales Month & Year

- Avg_Temp : Average temperature in that city for that month & year.

- Avg_Demand : Internal metric used to track demand

- Fuel : Fuel type. There are three wood-based fuels, Pellet, Briquette & Firewood. You can read about fuels [here](https://biofuelz.com/), as an example

- Sales : Aggregate sales that month-year

  
  

**Example of Wood Fuel**

  

Pellet

![](https://biofuelz.com/wp-content/uploads/2018/08/woodpellets-img1.png)

# Project Goals

The goals for this project are:

  

- Gather insights from data analysis that the fuel supplier can use

- Create accurate, robust forecast for CY2020 for the three fuels

- The supplier is more interested in forecast accuracy than forecast uncertainty, so forecasting models will be evaluated for accuracy rather than uncertainty

- The supplier is a small company without a DS team and would like to use simpler models that he can use, debug, maintain easily.

- Provide metrics to detect data drift

- Build a fair and inclusive ML solution

- Use Azure ML Service to deploy the model locally & use Power BI to consume the model & track forecast

  
  

# Deployment Architecture

  

The supplier uses Power BI in his company for data analytics and distributing enterprise reports in the company. Our solution uses Power BI for ETL (Extract, Transform, Load) from the enterprise data sources in the company. Power BI dataflow is created to transform the data in a format suitable for model building. The dataflow exports the *.csv file on a set schedule to Azure Data Lake Storage. Machine Learning model is built & tracked in Azure ML service in a locally deployed docker container. The scoring script runs the models, creates the forecast and saves it as a *.csv file back to the Azure Data Lake. Forecast is imported in Power BI and the sales forecast is distributed to everyone in the company using Power BI. The dashboard also includes *Data Drift* metric which alerts the user if the data structure has changed.

  

**Technologies Used:**

- Python

- Azure ML Service

- Azure Data Lake Storage

- Power BI

- Docker Container

  

!["Architecture](https://raw.githubusercontent.com/pawarbi/datasets/master/forecast_workflow.jpg)

  

## Model Training Process


For model building we exclusively focused on using classical forecasting methods because:

- We have only 60 observations for each fuel type. ML models typically need large number of observations to capture non-linear behavior in the time series

- ML models work well when many features are present. In our case, only temperature data is available. The problem with using temperature data for forecasting is that to predict the future sales, we will also need to first predict the future temperatures, which is not an easy task and can lead to inaccurate results

- The fuel supplier wants to maintain & update the forecast model without a dedicated data science team. So our focus is to create a modeling methodology that the supplier can easily follow, on his own, and train/update model in the future

- Unlike other prediction problems where a model is trained once and data drift is monitored, in time series forecasting the model has to be trained again when new data are available. Our goal is to not only provide forecast algorithm but to provide a framework for future models.
  
  

### Individual Forecasting Models Used

  

- SES : Single Exponential Smoothing

- Holt : Holt's

- Damp : Damped Trend Method

- HW1 : Holt Winter

- Naive : Seasonal Naive

- ETS1 : Error Trend Seasonality

- ETS_BoxCox : Error Trend Seasonality with BoxCox Transformation

- SARIMA : Seasonal Auto Regressive, Integrated, Moving Average

- SARIMAX : Seasonal Auto Regressive, Integrated, Moving Average - Exogenous

- Prophet : Facebook Prophet

- CombSHD : Combination SHD

- BaggedETS : Bagged ETS model

- Median : Median of all the above models

  

### Novel Modeling Approaches

Along with the classical forecasting methods (Exponential smoothing, SARIMA, SARIMAX, Facebook Prophet, Naive), we created python implementation of two new forecasting techniques - combshd & baggedETS.

  

#### CombSHD


CombSHD stands for *Combination- Single ExponentailSmoothing*, Holt's Method and Damped Method. While the method is not new, it's not available in Python to our knowledge. This method, although simple, has been shown to work better than even complex machin learning algorithms. As can be seen from the plot below from the M3 competition results, CombSHD, despite being simple, did as well as the complex methods and better than most other forecasting methods.

  

In this method, three different forecasts are created and are combined to create the final forecast.

  

Below plot shows (statistical) rank vs method. The lower the rank, the more accurate the method

  

![](https://ars.els-cdn.com/content/image/1-s2.0-S0169207004000810-gr1.jpg)

#### BaggedETS

  

BaggedETS is 'Bagging Exponential SMoothing' method. This is an ensemble forecasting method proposed by Prof. Rob Hyndman. The theory of this method can be read [here](https://robjhyndman.com/papers/BaggedETSForIJF_rev1.pdf). In a nutshell, to create a forecast:

  

- Time Series is decomposed using STL decomposition

- Residuals from the decomposition are bootstrapped

- Recombined many times to obtain new series

- Estimate a model for each bootstrapped series

Prof. Hyndman shows in his paper that this method also has shown to be effective than many other forecasting methods and actually creates much more robust forecast.

To our knowledge, there is no Python implementation of this method and we developed it for our project.

  
  

#### Ensemble Forecasts

  

Individual forecasts from 10 different forecasting methods are created and are combined, resulting in 1100+ different forecast models. Best among them is chosen based on :

- Error Metric : RMSE (since the data are free of outliers). The smaller the RMSE the better

- Parsimonious (i.e simple model with fewer parameters)

- Robust methods with previously known combinations are picked to avoid spurious fitting

  


## Modeling Procedure:

  

1. Create a train/validation/test split. First 75 months are used for training and validation. The last 3 months are held out.

2. Expanding window cross-validation on Holt Winter's method is run to find parameters with smallest RMSE

2. Order of SARIMA is obtained using pmdarima

3. 10 Different forecasts are obtained using various individual methods discussed below

4. All possible forecast combinations (1100+ models) are obtained by combining individual forecasts and averaging using median.

5. Final forecast combination is chosen for each fuel type by calcultaing RMSE for validation set (9 months)

6. Final model is created by training on all of the available dataset to create dynamic forecast

  
  

## Final models selected:

  

1. Pellet: combshd

2. Briquette: combshd + SARIMA

3. Firewood : Holt + Prophet

  

In our modeling we found that despite avg_temperature has strong relationship with the Sales, SARIMAX model which uses temperature as an exogenous variable, did poorly compared to most methods, except for Pellet. We tried using temperature as a regressor in Prophet model too, but results did not show improvement over Prophet model. The reason could be that because weather pattern doesn't change much, monthly seasonality already captures the variation. Temperature doesn't provide any additional information to the model that can explain sales.

  
  

## Data Drift

  

Machine Learning models and statistical models are built on the assumption that the structural pattern in the data will remain the same. When the underlying process and the distribution change, these assumptions are no longer valid and existing model can lose its predictive power. This is called as **Data Drift**. Some of the reasons for data drift in sales forecasting could be:

1. Change in Market conditions

2. Change in macro economics

3. Change in business processes

4. Change in data gathering/logging process

5. Change in customer prefernces etc.

  
  

While there are many advanced supervised and unsupervised learning methods to detect data drift, in our case since the data size is small and there is only one feature (Sales), we will define the temporal structure of the time series and use it to detect changes in the data. We are not aware of any published literature on this topic that is applicable to our case, so this is a proposed method and we plan to study it further in the future.

  

We define the temporal structure by :

  

1.  **Seasonality** : Power Spectral Density Analysis of the time series is done and peak frequency is extracted to detect seasonality

2.  **Auto-correlation Function (ACF)** : Classical as well as ML methods learn from the autocorrelation structure presnet in the data. We find the dominant lags based on ACF

3.  **Coefficient of Variation (CV)**: CV is calculated by dividing standard deviation of the time series by the mean. CV gives an indication of the variation in the data. If the CV varies significantly compared to historical data, it can be an indication of data drift.

  

#### Procedure for Detecting Data Drift

  

1. Run the *datadrift.py* python script on the data used for modeling training and forecasting

2. Run the same script, using same parameters, on newly obtained time series

3. If any of the three data drift metrics are different, we recommend performing EDA again and then fitting new model based on the model training procedure laid out above.

4. Power BI dashboard automatically will detect if the new data are within +/- 10% of the data drift metric and will display if any data drift has been detected

  

## Fairness

  

In our case the data & deployed solution are proprietary and will be used within the company and hence many of the fairness principles are not directly applicable. Still we have made every effort to design our solution while keeping the AI Fairness principles in mind. Particularly:

  

1.  **Reliability**: We created our own data drift metric and displayed it on the dashboard to ensure forecast results are reliable.

2.  **Transparency & Accountability**: Additional information about the modeling process and accuracy are shown on the dashboard, along with the contact information should the user need any additional information about the forecast process/accuracy.

3.  **Inclusiveness**: The final solution will be used by everyone in the company. According to an estimate, 10% of world's male population suffers from color blindness. Keeping this in mind, we chose the color pellet for our dashboard to make it colorblind-friendly by using high contrast colors. We have also used symbols , texture (dotted lines, indicators etc.) in accordance with colorblindness accessibility requirements. We made changes to the dashboard by using the results from [Color Blindness Simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/)

  
  
  
  
  

```python

  

```
