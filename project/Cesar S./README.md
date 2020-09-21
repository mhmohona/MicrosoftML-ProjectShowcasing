<h1 align='center'>  Time Series Regression - Gold Value Prediction </h1>
<p align="center">
  <img src="https://github.com/HSFI/Gold_Serie_Analysis/blob/master/images/ml%20azure.jpg">
</p>

### Authors:

Team **#Sg_Spanish**: </br>

1. Íñigo Lejarza(@Íñigo Lejarza) </br>
2. Cesar Schiraldi(@Cesar S.) </br>
3. Armando Medina(@ketcx)</br>
4. Jacqueline Susan Mejía (@susyjam) </br>
5. Fernando Terrazas(@Fernando T) </br>
6. Stanley Salvatierra(@Stanley Salvatierra) </br>

Based on: https://github.com/kittinan/predict-gold-price

## Summary

The objective of this project is to create a model that will help in predicting future market values of Gold, using the resources learnt in the Azure Machine Learning Scholarship.

## Introduction

This project will require a model capable of predicting values for future times. Because of that the required model will be a Regressor, which will be trained with previous values of the serie.

## Data Collection

For our project we found a complete dataset with Gold Prices in the web site of World Web Council (https://www.gold.org/).  
The information available in that site includes price of Gold in many currencies, and the values per day, per month, and per year.

## Dataset Preparation

In our project we took a subset of the dataset available.
We worked with the Daily Time Series of Gold in U.S. dollars (USD) from 2017 to 2020.
Our working dataset was the file [Precio_2017_2020.csv](Precio_2017_2020.csv).  
That dataset was normalized using a MinMaxScaler() from sklearn.preprocessing.

## Model Selection

After testing many different approachs, we obtained the best approach by using a Long-Short Temr Mempory Neural Network.

## Train/Test Data Splitting

For this project we splitted the Dataset with the following criteria:  
Train Data: 2017-01-02 to 2020-04-16 [859 points]  
Test Data: 2020-04-16 to 2020-08-28 [ 96 points]

## Implementation using a Jupyter Notebook

The project was implemented using a Jupyter notebook, which is available in current repository.

A screenshot of the Jupyter Notebook running locally on our laptops:  
![Jupyter](/images/Jupyter_local_02.png)

## Implementation in Azure - Notebook

As a second step the project was implemented in Microsoft Azure Machine Learning, using a Notebook.
A screenshot of the Azure ML Notebook running on the cloud:  
![Notebook](/images/Jupyter_Azure_02.png)

## Implementation in Azure - Designer (preview)

As a third step we implemented some steps of the model using Azure ML Designer (preview).
A screenshot of the Data Flow Pipeline running on the cloud:
![Designer](/images/Pipeline_Azure_01.png)

## Model Evaluation

The proposed model when evaluating its performance against the test data, made a very good prediction of the future values, as can be seen from the following plot:  
![Results](/images/Prediction_01.png)

## Model Conclusions

The purpose of this project was to test the viability of predicting future values of a Time Series, usign Azure Cloud based tools.
**The model was able to accurately properly predict future values, both using local processed resources or cloud based Azure resources.**

## Project Conclussions

This model can be used to predict other variables with higher social impact. We selected to work with the Gold price values because of its easyness and availability.
