import pandas as pd

DATA_PATH = "data/Music Info.csv"
OUTPUT_PATH = "data/cleaned_data.csv"


def clean_data(data):
    return (
        data
        .drop_duplicates(subset="spotify_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data):
    return (
        data
        .drop(columns=["track_id", "name", "spotify_preview_url"])
    )


def main():
    # Read the raw dataset
    data = pd.read_csv(DATA_PATH)

    # Clean the data
    cleaned_data = clean_data(data)

    # Save the cleaned data
    cleaned_data.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()