## NAME:
News & information of Covid-19 analysis, visualization and detection by Logistic Regression, MultinomialNB & LinearSVC.

**Author** : Anik Chowdhury 


## DATASET

![DatasetImage](https://drive.google.com/uc?export=view&id=1PFohvKYHat44hy9XSQ45UBZnGkuIjW4p)


In this project, i scraped the news and information available on social sites like Facebook, Twitter, Lead Stories, Poynter, FactCheck.org, Snopes and EuVsDisinfo. I modified the raw data using NumPy and Pandas. The dataset contains title, text, source, label. There are four class of the label: True, Fake, Misleading and Explanatory. From this dataset I have trained a model that can detect the class of label of information if it is true, fake, misleading or explanatory.

The purpose of this project was to see how far I could get in creating Covid-19 related news & information classification and what insights could be drawn from that, then used towards a better model.

I have used matplotlib, plotly and seaborn for plotting and visualization of data. 

I have used 3 machine learning algorithms to analyze and detect information of covid-19. They are:
1. Logistic Regression
2. Multinomial Naive Bayes Classifier
3. LinearSVC Classifier

For both **logistic regression** and **multinomial naive bayes** classifier, the title of the dataset is preprocessed through stopword removal, lemmatization, tokenization. After that the data is splitted and fitted for prediction. I have used 4 classes i.e. **FAKE, TRUE, MISLEADING, Explanatory**. Logistic Regression performs well but multinomial naive bayes doesn't.

Later i analyzes the number & percentage of **Stop words, Proper Noun, Capital Letter in title, VBG (Verb, gerund or present participle) & Negation words in Title**. 

By analyzing these feature, i have found that: 

**Proper Noun**: 
Fake & misleading news have more proper nouns. Apparently the use of proper nouns in titles are very significant in differentiating fake & misleading from rea news.
Overall, these results suggest that the writers of fake & misleading news are attempting to attract attention by using all capitalized words, and squeeze as much substance into the titles as possible by skipping stop-words and increase proper nouns. 
Here is an example: 
_Fake news: "FULL TRANSCRIPT OF “SMOKING GUN” BOMBSHELL INTERVIEW: PROF. FRANCES BOYLE EXPOSES THE BIOWEAPONS ORIGINS OF THE COVID-19 CORONAVIRUS"_

_Misleading news: ' The US uses the new coronavirus to put in place global control. They are going to inject nano-chips during vaccination, to control the foreign economies affected by COVID-19, and to govern those countries.'_

_Real news: "Why outbreaks like coronavirus spread exponentially, and how to 'flatten the curve'"_

_Explanatory news: ' A new outbreak pandemic of hantavirus is coming from China.'_

On the other hand there is not much difference between real and explanatory news.


**Stop Words**:
Fake & misleading news have less percentage of stop-words than those of real & explanatory news. 



**Capital Letter in title**:
On average, fake news have way more words that appear in capital letters in the title.This makes us to think that fake news is targeted for audiences who are likely to be influenced by titles. On the other-side real news have very few capital letters in text than fake and misleading news. Explanatory news have few capital letters among all.


**VBG (Verb, gerund or present participle)**:
Fake & misleading news have more VBG (Verb, gerund or present participle) than real & explanatory news.

Later i used those features to predict using LinearSVC classifier. 

I have attached images of code and output of Confusion matrix and plot of both training and test result, so that the performance of the model of above mentioned all three algorithms could be understood properly. 
  

## USED LIBRARIES AND IMPLEMENTED KNOWLEDGE:

1.	Pandas
2.	Matplotlib
3.  Seaborn
4.	Sickit learn
5.  NLTK

implemented Knowledge from course work
1.	Data wranggling and preprocessing
2.	Reducing overfitting, underfitting
4.	Machine learning Algorithm
5.	Model Evaluation, Confusion Matrixes 
6.	Plotting predicted result


All the codes and their outputs are attached. **4. Multinomial_naive_bayes_based_detection** folder contains all the images of code and output of multinomial naive bayes based detection. **5. LinerSVC_based_detection** folder contains images of codes & output related to linearSVC based detection. And **6. Logistic_Regression_based_detection** folder contains images of codes & output related to logistic regression based detection. 