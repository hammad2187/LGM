import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix

# Load the dataset
data = pd.read_csv("D:\songs.csv")

# Handle missing values by filling with an empty string for text-based columns
text_columns = ['artist_name', 'composer', 'lyricist', 'language']
data[text_columns] = data[text_columns].fillna('')

# Explicitly convert non-string values to strings
for column in text_columns:
    data[column] = data[column].astype(str)

# Combine text-based features into a single feature
data['text_features'] = data[text_columns].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text_features'])

# Convert the TF-IDF matrix to a compressed sparse row format for efficient computation
tfidf_matrix_csr = csr_matrix(tfidf_matrix)

# cosine similarities between songs
cosine_similarities = linear_kernel(tfidf_matrix_csr, tfidf_matrix_csr)

# mapping of song_id to song index
song_id_to_index = pd.Series(data.index, index=data['song_id']).drop_duplicates()

# Function to get song recommendations based on a given song
def get_recommendations(song_id, cosine_sim=cosine_similarities):
    idx = song_id_to_index[song_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    return data['song_id'].iloc[song_indices]

# recommendations for a song
song_id = "CXoTN1eb7AI+DntdU1vbcwGRV4SCIDxZu+YD8JP8r4E="
recommendations = get_recommendations(song_id)
print("Recommended Songs:")
print(data.loc[recommendations]['song_id'])
