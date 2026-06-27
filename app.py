import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd


# transformed data path
transformed_data_path = "data/transformed_data.npz"

# cleaned data path
cleaned_data_path = "data/cleaned_data.csv"


# load the data cleaned one
data = pd.read_csv(cleaned_data_path)

# load the trtansformed data
transformed_data = load_npz(transformed_data_path)

st.title("Welcome to Hybrid Recommender App")

st.subheader("Enter the name of the song and the recommender will suggest similar songs")

# Text input
song_name = st.text_input('Enter a song name')
st.write('You entered :',song_name)
# lowercase the input
song_name = song_name.lower()

# k recommendations
k = st.selectbox('How many recommendation do you want?',[5,10,15,20],index=1)


# Button
if st.button("Get recommendations"):

    if (data["name"] == song_name).any():

        st.write(f"Recommendations for **{song_name.title()}**")

        recommendations = recommend(song_name, data, transformed_data)

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