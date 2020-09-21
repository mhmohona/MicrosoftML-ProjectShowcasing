
# Music Recommendation System

**Team**    : Aarthi Alagammai , Anupriya Saraswat , Divyanshu Malik, Garima Sharma , Harshit Rai , Nidhi Singh, Snehal Lokesh 


**Introduction:**
Recommender System are widely used today in all most all the applications.The purpose of a recommender system is to suggest users something based on their interest or usage history.Two most ubiquitous types of personalized recommendation systems are Content-Based and Collaborative Filtering. Collaborative filtering produces recommendations based on the knowledge of users� attitude to items, that is it uses the �wisdom of the crowd� to recommend items. In contrast, content-based recommendation systems focus on the attributes of the items and give you recommendations based on the similarity between them.
We have created a Recommender sysem using Spotify
We have Scrapped dataset from [SPOTIFY ](http://spotify.com/) using our custom scraper, "Scrapify".
The Scrapped data is converted to as csv file and used for further processing.The dataset contains appromixately 11k observations

**Data Description:**

-name                : Name of the user

-artist              : Name of the artist

-danceability        : Ranges from 0 to 1

-key                 : Ranges from 0 to 11

-mode                : Ranges from 0 and 1

-instrumentalness    : Ranges from 0 to 1

-duration            : Duration of the song in minutes

-energy              : Ranges from 0 to 1

-loudness            : Float typically ranging from -60 to 0

-speechiness         : Ranges from 0 to 1

-acousticness        : Ranges from 0 to 1

-tempo               : Float typically ranging from 0 to 150

-liveness            : Ranges from 0 to 1

-valence             : Ranges from 0 to 1

-popularity          : Ranges from 0 to 100

-hollywood           : Hollywood song 1 | Bollywood song 0

# Project Goals
The goals for this project are: 

 -Scrap the website and collect the required data

 -Organise the data into a Structured format

- Gather insights from data analysis about the columns used

- Perform EDA and remove unwanted columns

 -Use the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two songs. Since we have used the vectors, calculating the Dot Product will directly give us the Cosine Similarity Score.

- Ouput the top 5 recommended songs

**Technologies Used:**
- Python
- Google Colab 
- Spotify API & custom scraper

## Model Training Process
For model building we exclusively focused on using collabrative filtering approach because:
   - We dont have any exclusive ratings of the users for the songs
   - We just have user prefernces 


```python

```