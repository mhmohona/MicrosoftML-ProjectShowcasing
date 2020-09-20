# EntityMatching

During the Azure Machine Learning Scholarship program, Lane posed a question on Slack asking how to go about doing entity matching using AI. There were varied responses from professionals who had faced similar problems, and the suggested solutions included Power BI (with about 90% accuracy as per @Sandeep Pawar), fuzzy matching with Levenstein distance, and concatenating the records to create a matrix and matching the similar words. None of these methods were deemed very successful but the posters did not believe that AI would give better results.

We decided to see for ourselves.

In this project, Zoé Goey & Lane endeavor to compare the Python Record Linkage Toolkit (PRLT) to DeepMatcher for entity matching. We use the DBLP-ACM dataset downloaded from the [DataBase Group Leipzig ](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution), which includes two tables with 2616 and 2294 rows, respectively. This project was done in Google Colab and targets clients who wish to develop an entity matching algorithm for a new product. The PRLT achieved a best score using the Naive Bayes classifier and only slightly underperformed DeepMatcher. Both were considerably better than the approach using fuzzy matching with Levenstein distance that we implemented. So we are inclined to say that machine learning does have added value in the domain of entity matching. 

Which machine learning  method is to be prefered is something that needs further investigation. DeepMatcher scored best, but the training effort was considerable and would probably be an overkill for easy data sets. Classical supervised machine learning gave comparable results in a fraction of the time needed by deep learning, and one of the unsupervised methods showed a very decent performance without requiring any labeling. So what to choose is probably dataset- and situation-dependent and would be a nice topic for a subsequent project.

To run the notebook (or in pacticular DeepMatcher), one needs a high-end machine with a good GPU and sufficient memory, so the easiest way to run it is on Google Colab. Just copy the entire contents of this directory into a directory named EntityMatching in the root of your Google Drive or correct the paths in the notebook so that the input files can be found.


