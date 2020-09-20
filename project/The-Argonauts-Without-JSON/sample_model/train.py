import model_dispatcher
import click
import joblib
import pandas as pd 
import os
from sklearn import metrics
from utils.split_data import split_data
import config


@click.command()
@click.option("--model", type=str, help="Enter the model")
@click.option("--neighbors", type=int, help="Number of neighbors")
def run(model, neighbors):
    file = None
    SEED = None
    df = pd.read_csv(config.MODEL_INPUT, usecols=["artists", "acousticness", "danceability", 
                                    "energy", "instrumentalness", 
                                    "liveness", "loudness", "speechiness", 
                                    "tempo", "valence", "popularity"])

    X_train, X_valid, _, _ = split_data(df, df.iloc[:,-1])

    X_train.to_csv("data/train_lookup.csv")
    X_valid.to_csv("data/valid_lookup.csv")

    X_train = X_train.iloc[:, 1:].values

    model = model_dispatcher.models[model](n_neighbors=neighbors, algorithm='ball_tree')

    model.fit(X=X_train)

    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"knn_.bin"))


if __name__ == "__main__":
    run()

    

    
