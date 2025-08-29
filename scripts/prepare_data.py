# Access a file
root = '../'
file_path = root + 'data/ml-100k/'

import pandas as pd

def load():
    ratings = pd.read_csv(file_path + "u.data", sep="\t", names=["user_id","movie_id","rating","ts"])
    movies_raw = pd.read_csv(file_path + "u.item", sep="|", header=None, encoding="latin-1")
    genre_cols = ["unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary",
                  "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
                  "War","Western"]
    movies = movies_raw[[0,1]].copy()
    movies.columns = ["movie_id","title"]
    movies[genre_cols] = movies_raw.iloc[:,5:24].values
    movies["genre_text"] = movies[genre_cols].apply(lambda r: " ".join([g for g,v in zip(genre_cols, r) if v==1]), axis=1)
    return ratings, movies, genre_cols

if __name__ == "__main__":
    ratings, movies, genre_cols = load()
    print(ratings.head())
    print(movies.head())
    ratings.to_parquet(root + "models/ratings.parquet", index=False)
    movies.to_parquet(root + "models/movies.parquet", index=False)
