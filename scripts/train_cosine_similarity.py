# training/02_train_content.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

root = '../'
movies = pd.read_parquet(root + "models/movies.parquet")

corpus = movies["genre_text"]
tfidf = TfidfVectorizer(norm=None, stop_words="english", min_df=1)
X = tfidf.fit_transform(corpus)

# Pre-compute item-item similarity (or defer to runtime for memory)
sim = cosine_similarity(X, X).astype(np.float32)

# movie dictionary
movie_dict = {title: idx for idx, title in enumerate(movies['title'])}
joblib.dump(movie_dict, root + "models/movie_dict.joblib")

joblib.dump(tfidf, root + "models/tfidf.joblib")

np.save(root + "models/content_sim.npy", sim)
