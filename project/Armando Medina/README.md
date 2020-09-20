<h1 align='center'>  Fake news classifier using Machine Learning </h1>
<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/ml%20azure.jpg">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/fakenews.jpeg" width=600>
</p>

Detection of fake news online is important in today's society as fresh news content is rapidly being produced as a result of the abundance of technology that is present. Fake news is nothing new. But, what is new is how easy it's become to share information – both true and false – on a massive scale.

Social media platforms allow almost anyone to publish their thoughts or share stories to the world. The trouble is, most people don't check the source of the material that they view online before they share it, which can lead to fake news spreading quickly or even "going viral."

Our work will consist of building a fake news classifier using machine learning.
### Authors: 
Team **#Sg_Spanish**: </br>
  1. Armando Medina(@ketcx)</br>
  2. Jacqueline Susan Mejía (@susyjam) </br>
  3. Fernando Terrazas(@Fernando T) </br>
  4. Stanley Salvatierra(@Stanley Salvatierra) </br>
  5. Cesar Schiraldi(@Cesar S.) </br>
  6. Íñigo Lejarza(@Íñigo Lejarza) </br>

## Table of Contents

<details open>
<summary>Show/Hide</summary>
<br>

1. [Introduction: Business Problem](#introduction)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Analysis](#analysis)
5. [Applying Designer](#applyingDesigner)
6. [Results and Discussion](#results)
7. [Conclusion](#conclusion)
</details>

## Introduction: Business Problem


<a name="#introduction"></a>
Typically spread over social media and traditional news outlets, misinformation remains rampant through the use of clickbait headlines and polarizing content.
With recent world events, we’ve seen how much impact the news has on our lives. From understanding what is happening surrounding the pandemic to the movement of the stock market, I know that I rely heavily on the news, and I’m sure everyone else does too. However, it is often difficult to distinguish between articles with false information and those providing real, fact-checked news. Given that companies such as Facebook and Twitter deploy algorithms to ensure that people are receiving the right, correct information on their feeds, I wanted to explore utilizing Natural Language Processing and text analysis to build a fake news classifier.
<br>

## Data


<a name="#data"></a>
This dataset contains a list of articles considered as "fake" news 
For this example we are based on the Kaggle data https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset#True.csv

<br>


## Methodology

<a name="#methodology"></a>
Sequential Model and Bi-Directional RNN and LSTM
<br>

## Exploratory Data Analysis

<a name="#analysis"></a>

<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/isfake.png">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/4ml.png">
</p>

<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/6ml.png">
</p>
In the data you can see the words "Fake"

<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/fake.png">
</p>

In the data you can see the words "Real"
<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/real.png">
</p>
<br>

## Applying Designer

<a name="#applyingDesigner"></a>
We used the Designer inspired by the laboratory of the course on text classification to evaluate how the result would be obtained with the Two-Class Logistic Regression algorithm

<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/3ml.jpeg">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/4ml.jpeg">
</p>
<br>

## Results and Discussion
<a name="#results"></a>

If the predicted value is > 0.5 it is real else it is fake But in our example the Model Accuracy is 0.9985152190051967

### Confusion matrix

<p align="center">
  <img src="https://github.com/ketcx/fake-news-classifier/blob/master/Images/matrix.png">
</p>
<br>

## Conclusion

<a name="#conclusion"></a>
<summary>Show/Hide</summary>
<br>

