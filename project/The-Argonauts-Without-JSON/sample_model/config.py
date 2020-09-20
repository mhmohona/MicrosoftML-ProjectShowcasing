import os

K_NEIGHBORS = 10

MODEL_OUTPUT = "{}/{}".format(os.path.abspath("."), "models/knn_.bin")

MODEL_INPUT = "{}/{}".format(os.path.abspath("."), "data/data_by_artist.csv")

LOOKUP_TABLE = "{}/{}".format(os.path.abspath("."), "data/train_lookup.csv")

MASTER_TABLE = "{}/{}".format(os.path.abspath("."), "data/data.csv")