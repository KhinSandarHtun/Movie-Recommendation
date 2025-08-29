import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------
# Paths
# ------------------------
root = "./"  # adjust if needed

# ------------------------
# Load precomputed models and data
# ------------------------
@st.cache_data
def load_models():
    movies = pd.read_parquet(root + "models/movies.parquet")
    movie_dict = joblib.load(root + "models/movie_dict.joblib")
    content_sim = np.load(root + 'models/content_sim.npy')
    return movies, movie_dict, content_sim

movies, movie_dict, content_sim = load_models()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Movie Recommendation", page_icon="🎬", layout="wide")
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Select a movie to get recommendations based on genres.")

movie_name = st.selectbox("Choose a movie:", movies['title'].values)

num_rec = st.slider("Number of recommendations:", min_value=1, max_value=50, value=5)
# ------------------------
# Display movie genre
# ------------------------
#if movie_name in movie_dict:
#    movie_index = movie_dict[movie_name]
#    genre = movies.iloc[movie_index]["genre_text"]
#    st.subheader(f"Genres: {genre}")
# ------------------------
# Recommendation Function
# ------------------------
def get_similar_movies(movie_name, num=5):
    if movie_name not in movie_dict:
        return []

    movie_index = movie_dict[movie_name]
    distances = content_sim[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num + 1]
    selected_movie = movies.iloc[movie_index]
    selected_genre = selected_movie["genre_text"]

    rec_movies = []
    for i, (idx, score) in enumerate(movie_list):
        movie = movies.iloc[idx]
        rec_movies.append({
            "Rank": i+1,
            "Title": movie['title'],
            "Genres": movie['genre_text'],
            "Score": round(score, 4)
        })
    return selected_genre,rec_movies

# ------------------------
# Display Recommendations
# ------------------------
if st.button("Recommend"):
    selected_genre, recommendations = get_similar_movies(movie_name, num_rec)
    if recommendations:
        st.subheader(f"Movie Name :'{movie_name}'")
        st.subheader(f"**Movie genre:** {selected_genre}")
       # st.markdown("---")
        st.subheader("Recommended movies:")
        rec_df = pd.DataFrame(recommendations)
        #st.table(rec_df)
        st.dataframe(rec_df, hide_index=True)

    else:
        st.warning("Movie not found in database.")
