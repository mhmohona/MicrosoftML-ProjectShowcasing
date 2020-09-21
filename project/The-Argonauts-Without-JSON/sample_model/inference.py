import joblib
import pickle
import os
import config
import pandas as pd
import click


def load_model_helper(file_path):
    if os.path.split(".")[-1] == "pickle":
        return pickle.load(open(file_path, 'wb'))
    
    return joblib.load(file_path)

def fetch_artist_columns(df, artist_list):
    return [artist for artist in df["artists"].to_list() for a in artist_list if a in artist]


class SpotifyRecommender:
    def __init__(self, model):
        self.model = model
    
    def _predict(self, arr, k=20):
        return self.model.kneighbors(arr, 
                                     n_neighbors=k, 
                                     return_distance=False)
    
    def create_playlist(self, arr):
       predictions = self._predict(arr)
       lookup_table = pd.read_csv(config.LOOKUP_TABLE)
       artist_list = lookup_table.iloc[predictions[0][1:], 1].to_list()
       master_table = pd.read_csv(config.MASTER_TABLE, usecols=["artists", "name", "popularity"])

       songs = master_table[master_table["artists"].isin(fetch_artist_columns(master_table, artist_list))]
       songs = songs.drop_duplicates(subset=["name"], keep="first")

       return [*songs[["artists", "name"]].sample(n=30).itertuples(name="Songs", index=False)]


@click.command()
@click.option("--artist_name", type=str, help="Enter the artist name.")
def main(artist_name):
    model = load_model_helper(config.MODEL_OUTPUT)
    spotify_recommender = SpotifyRecommender(model)
    df = pd.read_csv(config.MODEL_INPUT, usecols=["artists", "acousticness", "danceability", 
                                    "energy", "instrumentalness", 
                                    "liveness", "loudness", "speechiness", 
                                    "tempo", "valence", "popularity"])
    arr = df[df["artists"].isin([artist_name])].values[:,1:]
    
    playlist = spotify_recommender.create_playlist(arr)
    print(playlist)
    

if __name__ == "__main__":
    main()

    
