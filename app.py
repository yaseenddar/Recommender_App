import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from collaborative_filtering import collaborative_recommendation
from numpy import load



# cleaned data path
cleaned_data_path = "data/cleaned_data.csv"
# load the data cleaned one
songs_data = pd.read_csv(cleaned_data_path)

# transformed data path
transformed_data_path = "data/transformed_data.npz"
# load the trtansformed data
transformed_data = load_npz(transformed_data_path)
#  load the track ids
track_ids_path = "data/track_ids.npy"
track_ids = load(track_ids_path,allow_pickle=True)

filtered_data_path = "data/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

# interaction the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

st.title("Welcome to Hybrid Recommender App")

st.subheader("Enter the name of the song and the recommender will suggest similar songs")

# Text input
song_name = st.text_input('Enter a song name')
st.write('You entered :',song_name)
# lowercase the input
artist_name = st.text_input("Enter the artist name")
st.write('You entered: ',artist_name)

song_name = song_name.lower()
artist_name = artist_name.lower()

# k recommendations
k = st.selectbox('How many recommendation do you want?',[5,10,15,20],index=1)

filtering_type = st.selectbox("Select the type of filtering:",['Content-Based Filtering','Collaborative Filtering'])
# Button
if filtering_type == 'Content-Based Filtering':
    if st.button("Get Content-Based recommendations"):
        song_exists = (
        (songs_data["artist"] == artist_name) | (songs_data["name"] == song_name)
        ).any()
        if (song_exists):

            st.write(f"Recommendations for **{song_name}**")

            recommendations = content_recommendation(
                song_name,
                artist_name,
                  songs_data,
                 transformed_data)

            for ind, recommendation in recommendations.iterrows():

                recommended_song = recommendation["name"].title()
                artist_name = recommendation["artist"].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(
                        f"#### **{recommended_song}** by **{artist_name}**"
                    )
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")

                elif ind == 1:
                    st.markdown("### Next Up")
                    st.markdown(
                        f"#### {ind}. **{recommended_song}** by **{artist_name}**"
                    )
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")

                else:
                    st.markdown(
                        f"#### {ind}. **{recommended_song}** by **{artist_name}**"
                    )
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")

        else:
            st.error(f"We could not find '{song_name}' in our database. Please try another song.")

elif filtering_type == 'Collaborative Filtering':
    if st.button('Get Collaborative Recommendation'):
        song_exists = (
            (filtered_data["name"] == song_name) |
            (filtered_data["artist"] == artist_name)
        ).any()
        st.write(song_exists)
        if (song_exists):
            st.write("Recommendations for",f"**{song_name}** by **{artist_name}** ")
            recommendations = collaborative_recommendation(song_name=song_name,
                                                            artist_name=artist_name,
                                                            track_ids=track_ids,
                                                            songs_data=filtered_data,
                                                            interaction_matrix=interaction_matrix)
            
            # Display Recommendation
            for ind, recommendation in recommendations.iterrows():
                song_name = recommendation["name"].title()
                artist_name =  recommendation['artist'].title()

                if ind == 0:
                    st.write("## Current Playing")
                    st.markdown(f"### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                elif ind == 1:
                    st.markdown("### Next Up ")
                    st.markdown(f"#### {ind}, **{song_name}** by {artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('----')
                else:
                    st.markdown(f"#### {ind}, **{song_name}** by {artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
        else:
            st.error(f"We could not find '{song_name}' in our database. Please try another song.")
            