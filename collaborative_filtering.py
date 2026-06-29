import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix,save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# set paths
# output paths

track_ids_save_path = "data/track_ids.npy"
filtered_data_save_path = "data/collab_filtered_data.csv"
interaction_matrix_save_path = "data/interaction_matrix.npz"

# input paths
songs_data_path = "data/cleaned_data.csv"
user_listening_hsitory_data_path="data/User Listening History.csv"


def filter_songs_data(songs_data:pd.DataFrame,track_ids,save_df_path:str)->pd.DataFrame:
    """
    Filter the songs data for the given track ids

    """

    # filter data based o nteh the track ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    # rest index
    filtered_data.reset_index(drop=True,inplace=True)
    # save the data
    save_pandas_data_to_csv(filtered_data,save_df_path)
    
    return filtered_data
    
def save_pandas_data_to_csv(data:pd.DataFrame,file_path:str)->None:
    """
    Save the sparse matrix to a npz file
    """

    data.to_csv(file_path,index=False)

def save_sparse_matrix(matrix:csr_matrix,file_path:str)-> None:
    """
    SAve the sparse matrix to a npz file
    """

    save_npz(file_path,matrix)

def create_interaction_matrix(history_data:dd.DataFrame,track_ids_save_path,save_matrix_path)->None:
    # make a copy of data
    df = history_data.copy()

    # convert the playcount into float
    df["playcount"] = df["playcount"].astype(np.float64)

    # convert string column to categorical
    df = df.categorize(columns=['user_id','track_id'])

    #  Convert user_id and track_id to numeric indices
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    #  get the list of track_ids
    track_ids = df['track_id'].cat.categories.values
    
    np.save(track_ids_save_path,track_ids,allow_pickle=True)

    # add the index columns to the dataframe
    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping
    ) 

    # create teh interaction matrix
    interaction_matrix = (
    df.groupby(["track_idx", "user_idx"])["playcount"]
      .sum()
      .reset_index()
      .compute()
)

    # get teh indees to form sparse matrix
    row_indices = interaction_matrix["track_idx"]
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']

    # get the shape of the sparse matrix
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()

    # craete the saprse matrix
    interaction_matrix = csr_matrix((values,(row_indices,col_indices)),shape=(n_tracks,n_users))

    # save the sparse matrix
    save_sparse_matrix(interaction_matrix,save_matrix_path)

def collaborative_recommendation(
    song_name,
    artist_name,
    track_ids,
    songs_data,
    interaction_matrix,
    k=10
):

    # Normalize input
    song_name = song_name.strip().lower()
    artist_name = artist_name.strip().lower()

    # Find the song
    song_row = songs_data.loc[
        (songs_data["name"].str.lower() == song_name) |
        (songs_data["artist"].str.lower() == artist_name)
    ]

    if song_row.empty:
        raise ValueError(f"'{song_name}' by '{artist_name}' not found.")

    # Get track id
    input_track_id = song_row.iloc[0]["track_id"]

    # Find row index in interaction matrix
    matches = np.where(track_ids == input_track_id)[0]

    if len(matches) == 0:
        raise ValueError(f"Track ID '{input_track_id}' not found in track_ids.")

    input_index = matches[0]

    # Get interaction vector
    input_vector = interaction_matrix[input_index]

    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_vector, interaction_matrix).flatten()

    # Remove the song itself
    similarity_scores[input_index] = -1

    # Get top-k indices
    recommendation_indices = np.argsort(similarity_scores)[-k:][::-1]

    # Convert indices back to track_ids
    recommendation_track_ids = track_ids[recommendation_indices]

    # Create scores dataframe
    scores_df = pd.DataFrame({
        "track_id": recommendation_track_ids,
        "score": similarity_scores[recommendation_indices]
    })

    # Merge with songs dataframe
    recommendations = (
        songs_data.merge(scores_df, on="track_id")
        .sort_values("score", ascending=False)
        .drop(columns=["score"])
        .reset_index(drop=True)
    )
    return recommendations

def main():
    # load the hsitory data
    user_data = dd.read_csv(user_listening_hsitory_data_path)

    # get the ubique track ids
    unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()

    # filter the songs data
    songs_data = pd.read_csv(songs_data_path)
    filter_songs_data(songs_data,unique_track_ids,filtered_data_save_path)

    # create the interaction matrix
    create_interaction_matrix(user_data,track_ids_save_path,interaction_matrix_save_path)
if __name__== "__main__":
    main()

