import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz

from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from category_encoders import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# cols to transform
#  cols to transform

frequency_encode_cols = ["year"]
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
min_max_scale_cols = ['danceability','energy','speechiness',"acousticness","instrumentalness","liveness","valence"]
standard_scale_cols = ["duration_ms","loudness","tempo"]

def train_transformer(data):
    # transform the data
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode",CountEncoder(normalize=True,return_df=True),frequency_encode_cols),
        ("ohe",OneHotEncoder(handle_unknown="ignore"),ohe_cols),
        ("tfidf",TfidfVectorizer(max_features=85),tfidf_col),
        ("standard_scale",StandardScaler(),standard_scale_cols),
        ("min_max_scale",MinMaxScaler(),min_max_scale_cols),
])
    
    transformer.fit(data)

    joblib.dump(transformer,"transformer.joblib")

def transform(data):
    transformer = joblib.load("transformer.joblib")

    # transform the data
    transformed_data = transformer.transform(data)
    return transformed_data

def save_transformed_data(transformed_data,save_path):
    save_npz(save_path,transformed_data)
    

def calculate_similarity(input_vector,data):
    # we will justhave the input and data in vector form tfidf 
    # and we find the similarity between them using cosignsimilarity

    similarity_scores = cosine_similarity(input_vector,data)
    
    # The result is a 2D array, so we take the first (and only) row
    similarity_scores = similarity_scores.flatten()

    # # Create a dataFrame form them and send them
    # similarity_df = pd.DataFrame([
    #     'song_index':data.index,
    #     'artist':data['artist'],
    #     'year',data['year'],
    #     'similarity':similarity_score
    # ])

    # similarity_df = similarity_df.sort_values(by='similarity',ascending=False)

    # # display the top 10 to the user
    # display(similarity_df.iloc[:10])

    return similarity_scores    
import numpy as np

def recommend(song_name, songs_data, transformed_data, k=10):

    song_name = song_name.lower()

    song_row = songs_data[songs_data["name"] == song_name]

    if song_row.empty:
        return None

    song_index = song_row.index[0]

    input_vector = transformed_data[song_index]

    similarity_scores = calculate_similarity(
        input_vector,
        transformed_data
    ).ravel()

    # Sort by similarity
    top_k_songs_indices = np.argsort(similarity_scores)[::-1]
    # Remove the input song
    top_k_songs_indices = top_k_songs_indices[top_k_songs_indices != song_index]

    # Keep only k recommendations
    # top_k_songs_indices = top_k_songs_indices[:k-1]
    top_k_songs_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    # Get rows from DataFrame, not sparse matrix
    top_k_songs_rows = songs_data.iloc[top_k_songs_indices]

    return top_k_songs_rows[
        ["name", "artist", "spotify_preview_url"]
    ].reset_index(drop=True)

def test_recommendation(data_path,song_name,k=10):
        # convert song_name to lowercase
    song_name = song_name.lower()
    data = pd.read_csv(data_path)

    # clearn the data
    data_content_filtering = data_for_content_filtering(data)
    train_transformer(data_content_filtering)
    # transform the data
    transformed_data = transform(data_content_filtering)
    #save transformer data
    save_transformed_data(transformed_data,"data/transformed_data.npz")
    # filter out the song from the data
    song_row = data.loc[data["name"] == song_name]
    print(song_row)
    # get the index of the song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector= transformed_data[song_index].reshape(1,-1)
    # calculate the similaruty score
    similarity_scores = calculate_similarity(input_vector,transformed_data)
    
    top_k_songs_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    # get the top k osngs
    top_k_songs_rows = data.iloc[top_k_songs_indices]

    top_k_list = top_k_songs_rows[["name","artist","spotify_preview_url"]].reset_index()

    print(top_k_list)
if __name__ == "__main__":
    test_recommendation(CLEANED_DATA_PATH,"HIPS Don't Lie")