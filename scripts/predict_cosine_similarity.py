# Access a file
root = '../'

import numpy as np
import joblib
import pandas as pd

content_sim = np.load(root + 'models/content_sim.npy')
movie_dict = joblib.load(root + 'models/movie_dict.joblib')
movies = pd.read_parquet(root + "models/movies.parquet")

print(movie_dict)

def get_similar_movies(movie_name, num = 20):
  movie_index = movie_dict[movie_name]
  selected_movie = movies.iloc[movie_index]
  print(selected_movie['title'], selected_movie['genre_text'])
  distances = content_sim[movie_index]
  movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:num + 1]
  print('recommended movies:')
  for index, i in enumerate(movie_list):
    movie = movies.iloc[i[0]]
    print(index + 1, '.', i[0], movie["title"], movie['genre_text'], i[1])

movie = input('Enter movie name: ')
get_similar_movies(movie)