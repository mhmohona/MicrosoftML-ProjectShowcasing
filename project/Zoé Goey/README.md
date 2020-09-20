# EntityMatching

During the Azure Machine Learning Scholarship program, Lane posed a question on Slack asking how to go about doing entity matching using AI. There were varied responses from professionals who had faced similar problems, and the suggested solutions included Power BI (with about 90% accuracy as per @Sandeep Pawar), fuzzy matching with Levenstein distance, and concatenating the records to create a matrix and matching the similar words. None of these methods were deemed very successful but the posters did not believe that AI would give better results.

We decided to see for ourselves.

In this project, Zoé Goey & Lane endeavor to compare the Python Record Linkage Toolkit (PRLT) to DeepMatcher for entity matching. We use the DBLP-ACM dataset downloaded from [https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution), which includes two tables with . This project was done in Google Colab and targets clients who wish to develop an entity matching algorithm for a new product. The PRLT achieved a best score using the Naive Bayes classifier and only slightly underperformed DeepMatcher.

We performed feature engineering on PRLT as per the site's suggestions for dataset FEBRL4 but since DeepMatcher has built-in feature engineering functions, we just let it do the feature engineering automatically. Therefore, the feature vectors used were not the same.

Additionally, with PRLT, ECM and K-Means actually also reached a perfect score on the validation set, but did this without using the true links (unsupervised). It performed better than manual tuning and did so in a reasonable timeframe.
