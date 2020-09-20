# Algorithmic Trading

## Team:
@Lin C., @Somya Dwivedi, @Varada.B, @Garima Sharma, @Gayathri Rajan, @tejas, @Varsha.Gore, @Akhilesh, @Lamiae Hana

![github-small](https://content.fortune.com/wp-content/uploads/2019/10/GettyImages-1158402857.jpg?resize=750,500)

## Problem Statement
Our project showcase challenge is to take an algorithmic approach to stock trading. On Kaggle we found an open data set of the Dow Jones Industrial Average (DJIA),  covering eight years and also the top 27 Reddit news during that same time frame (citation Daily News for Stock Market Prediction). Our problem was to use machine learning models to explain and forecast. Specifically we applied various regression algorithms to the Dow Jones industrial average and used natural language processing (NLP) analysis on the Reddit news to see if it had a predictor influence on whether the Dow Jones close daily price went up, down or stayed the same.

## Significance of the Problem Statement 
The stock market is fundamentally a financial institution for companies to raise capital. A share in a stock represents an ownership of a fractional piece of the company. Wall Street, a collective of where stocks are bought, sold, and where capital is raised, is often thought of as having a ‘random walk’ (Citation: [Random Walk Down Wall Street](https://en.wikipedia.org/wiki/A_Random_Walk_Down_Wall_Street)). This means the underlying data set has statistical attributes that are defined as random. 
Financial markets are highly adversarial in nature and non-stationary. They change all the time, driven by political, social, economic or natural events. As human emotions and psychology are the driving force behind investment decisions and Wall Street.  
 
We believed the top Reddit news is a capable proxy for human emotions. An important component of a stock is it’s price and what it represents. It represents the sum of all future earnings, taken to its current net present value. We are using the DJIA as a proxy for the broad US stock market and it’s stock values. As an index, the DJIA has transparency or openess, meaning it’s value is reported continuously, during open trading hours, visible to anyone with the internet.  The approach of using algorithmic trading is often proprietary, tightly held secrets of an investment company. With the advent of machine learning, it would be interesting to see if the underlying data sets and machine learning models can be exploited to forecast and explain DJIA and reddit news’ influences. 

We were eager to try out our own hands with algorithmic trading using machine learning.
 
## Project Implementation 
The purpose of this project was for the model to be able to predict the stock price for the next trading day. We have implemented the project in two ways:
MS Azure ML implementation in Microsoft Azure ML Studio (Classic) and 
Python implementation in Jupyter notebook.

### Dataset
We used dataset from Kaggle : - [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews)

### Technologies 
Kaggle, Microsoft Azure ML Studio (Classic), Jupyter Notebook, PyTorch, GitHub

### MS Azure ML Implementation
We used Azure Machine Learning Studio (classic). Based on our knowledge gained from our class, we applied Azure to our Algorithmic trading, specifically using the regression analysis. We applied our key learnings to gain insights on how well our models worked to explain DJIA. We used machine learning models in Jupyter notebook with classic preprocessing of the data set, then using text NLP algorithms, time series forecasting (ARIMA) and Recurrent Neural Networks.


![Azure Experiment](https://github.com/VaradaB/algorithmic-trading/blob/master/Images/algotrade1.PNG?resize=500,700)


#### Model Architecture
We applied eight regression models: Bayesian linear regression, decision forest regression, boosted decision tree regression, fast forest quantile regression, neural network regression, ordinal regression and Poisson regression. We compared the output for each. See Figure 1. 

#### Model Training
We split the dataset 80% for training, and 20% for testing with the dates column. 

#### Model Metrics
We used RMSE with Bayes Linear Regression and Linear Regression gave the lowest RMSE.

![Azure Model Results](https://github.com/VaradaB/algorithmic-trading/blob/master/Images/algotrade2.jpg)


### Stock price forecasting using ARIMA (Auto Regressive Integrated Moving Average) model
The acronym of ARIMA stands for: AutoRegressive = the model takes advantage of the connection between a predefined number of lagged observations and the current one. 
Integrated = differencing between raw observations (eg. subtracting observations at different time steps). 
Moving Average = the model takes advantage of the relationship between the residual error and the observations.

The ARIMA model makes use of three main parameters (p,d,q). These are: P — Auto regressive feature of the model or number of lag observations D — Differencing order Q — Moving average feature of the model

ARIMA can lead to particularly good results if applied to short time predictions.

So we fit an ARIMA(5,1,0) model. This sets the lag value to 5 for autoregression, 
uses a difference order of 1 to make the time series stationary, and uses a moving average model of 0.

Output: RMS error: 166.74843134433715

### Stock price forecasting using LSTM (Long Short-Term Memory) model and GRU (Gated Recurrent Units) model

In this work we also explore the capability of Recurrent Neural Networks (RNN) to model the prediction across time. We train and evaluate two widely used RNNs - Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU). Both the models were trained using Adam optimizer and MSE loss function.

Output: Testing on scaled data in range (0,1)
Test accuracy on GRU (MSE loss): 0.0001475
Test accuracy on LSTM (MSE loss): 0.0001430

### NLP (Natural Language Processing)

NLP was used to understand how much the news impacted the stock market. The combination of numerical analysis and NLP analysis gave us fruitful results. The dataset that we obtained from Kaggle consisted of Top 25 news from Reddit WorldNews Channel. The NLP was carried out using three different models. This was done so that we could compare which model gave us better results.

#### Model 1

We concatenated all the news headlines of a single day into one. We used TF-IDF (Term Frequency–Inverse Document Frequency) vectorization to extract a feature vector. The classifier we used was a SVM (Support Vector Machine) with RBF (Radial Basis Function) kernel without optimization of hyperparameters.

Result: We obtained an accuracy of 53.9% and ROC-AUC of 0.52.

#### Model 2

In this model we used count vectorizer and logistic regression for label classification. n-grams are basically a set of co-occurring words within a given window and when computing the n-grams you typically move one word forward. We have tested with 1-gram, 2-gram and 3-gram.

Result:

![Accuracy for n-grams](https://github.com/VaradaB/algorithmic-trading/blob/master/Images/algotrade3.jpg)

The ROC-AUC of the model using 3-gram is 0.95.

#### Model 3

For this model we combined the numerical columns (open, high, low, close, volume, adj close) and the sentiment columns (subjectivity, objectivity, positive, neutral, negative) of the dataset.

Result: The Linear Discriminant Analysis score is 94.3% and ROC-AUC is 0.5.

#### Model 4

In this model we used XGBoost algorithm.

Result: The accuracy attained was 94.1% and ROC-AUC of 0.98.

## Innovation and Creativity 
This project is an innovative outcome of predicting the volatility of the stock market in terms of its opening, closing, high, low and average volume price combined with the sentiments of top news that affected it. The combination of numerical as well as NLP part has made this project show some fruitful results.
Our project has also supported the results done from Azure and through manual coding. 

## Impact & Potential 
As an index, the DJIA has transparency or openess, meaning it’s value is reported continuously, during open trading hours, visible to anyone with the internet. Obtaining real time price quotes allows for an orderly buying and selling of stocks. Based on algorithmic trading, companies will have more predictable valuation when going through initial public offerings and individual investors benefit from quantifying their investment risks. So when these technologies are deployed correctly, they will make investors more efficient so when people invest, they will invest following scientific approaches rather than making wild speculation.

## Responsible AI  
The predictions results from our models and analysis  will not be used for any wrong purposes. Models used here will perform safely and will be secure. As far as privacy is concerned the dataset we used is an open dataset on Kaggle, which is public. We have stacked stocks from different companies and our models are NOT biased towards any company's stock. All the fair practices have been used in the analysis  and all steps including coding are transparent. Explanations are given in detail and our models are not just black box one. Most importantly our team has included everyone in the team for tasks. Our team is an intentionally diverse team, where members have collaborated from four different time zones across the globe successfully.
