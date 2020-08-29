# Quote Finder
A flask web app that finds a quote on any given genre.

## Video
![Screenshot](quotr.gif)

## Dataset
The quotes are pulled from a CSV file obtained from [TheWebMiner](https://thewebminer.com/buy-famous-quotes-database) containing around 76,000 quotes across 117 genres, by over 11,000 authors.

## A quote lookup? Where's the AI?
Since there are only 117 genres in the quote library, if the user asks for a theme that doesn't exist as a label by itself the app will attempt to find the closest match. Sometimes it's good, sometimes not so good!

The closest match is identified by translating the requested genre and all existing genre labels into multi-dimensional vectors and locating the closest using a measure called cosine distance. The vectors were learned from text samples such that similar words have similar vector representations.

## Try the app
[http://quotr-ml.azurewebsites.net](http://quotr-ml.azurewebsites.net)

## Author
Aleem Juma


# Microsoft Azure Scholarship Project Showcasing Challenge
[Link to guidelines](https://docs.google.com/document/d/1p0rplg0ZrIFfBabY1WyhyVOxjVjxMORC3koV00rscAI/edit)

All criteria are fully met for a full 100% score!

### Using Azure for Implementation based on Course Material
The project is hosted in Azure, using an Azure App Service, connected for continuous integration and continuous deployment to [my github repository](https://github.com/scign/quotr).

### Innovation & Creativity
To my knowledge there is no similar service available on the internet. Through extensive Google searching I was unable to locate a quote search engine that uses ML/NLP to find similar topics.

### Project Implementation
The idea is fully implemented.

### Impact & Potential
Communication is a need in all situations. People communicate best when they find mutual common ground. Using the words of others - especially when those words are well known as established "quotes" helps people find that common ground. Having a handy quote finder for any topic is helpful to ease communication between people of diverse backgrounds. By scaling this idea, we help people communicate better leading to stronger, more lasting relationships. Ultimately as more and more people communicate better thanks to this tool, conflicts will reduce, wars will be averted, and world peace will finally become a reality.

### Responsible AI
* Privacy/Security - No user data is ingested or revealed through using this tool.
* Accountability - All quotes are attributed to their sources and the page contains a reference to this repository for full code accountability.
* Reliability/Safety - If a word is not found in the dataset or the model, the tool fails gracefully and selects a random topic for the user.
* Fairness - The tool provides quotes in English for maximum accessibility worldwide.
* Inclusiveness - The model is sourced from the spaCy en_core_web_md model, trained on OntoNotes 5 which is a diverse dataset to ensure reasonable inclusivess of ideas. Quotes are in English only which matches the model vocabulary and is the most widely spoken language in the world which further facilitates communication.
* Transparency - The page makes it clear that an AI model is involved in returning information ("AI-powered similarity result" is displayed), and shows potential other matches that can be selected for further refinement.


## Potential further improvements
* Show similar options even when a match is found in the database genres
* Expand quote database
* Add translation engine to translate quotes to other languages to improve accessibility
* Allow natural language input (e.g. "give me a quote to make me feel more positive")