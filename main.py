import spotipy
from spotipy.oauth2 import SpotifyOAuth

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pathlib
from keras.utils.vis_utils import pydot
from keras.utils.vis_utils import plot_model

import tensorflow_utils as tu

# -----

scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope,
                                               client_id='XXXX',
                                               client_secret='XXXX',
                                               redirect_uri='http://example.com', ))

playlists = sp.current_user_playlists(20, 0)

# Users Playlist
user_playlist = "Anu"

playlist_items = playlists['items']

playlist = [x for x in playlist_items if x['name'] == user_playlist]

playlist_id = playlist[0]['id']

playlist_tracks = sp.playlist(playlist_id)

# Getting track ids of selected playlist
track_id_list = list(map(lambda x: x['track']['id'], playlist_tracks['tracks']['items']))

# Getting audio features from spotify
audio_features_result = sp.audio_features(track_id_list)

print(audio_features_result)
# -----

# audio_features_result = [
#     {'danceability': 0.358, 'energy': 0.73, 'key': 0, 'loudness': -10.251, 'mode': 0, 'speechiness': 0.0657,
#      'acousticness': 0.792, 'instrumentalness': 0.0161, 'liveness': 0.177, 'valence': 0.115, 'tempo': 169.936,
#      'type': 'audio_features', 'id': '0TiZDNPU2t4STNJW4Qdj22', 'uri': 'spotify:track:0TiZDNPU2t4STNJW4Qdj22',
#      'track_href': 'https://api.spotify.com/v1/tracks/0TiZDNPU2t4STNJW4Qdj22',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0TiZDNPU2t4STNJW4Qdj22', 'duration_ms': 293510,
#      'time_signature': 4},
#     {'danceability': 0.735, 'energy': 0.677, 'key': 2, 'loudness': -4.979, 'mode': 1, 'speechiness': 0.093,
#      'acousticness': 0.0762, 'instrumentalness': 2.17e-05, 'liveness': 0.111, 'valence': 0.188, 'tempo': 100.584,
#      'type': 'audio_features', 'id': '77UjLW8j5UAGAGVGhR5oUK', 'uri': 'spotify:track:77UjLW8j5UAGAGVGhR5oUK',
#      'track_href': 'https://api.spotify.com/v1/tracks/77UjLW8j5UAGAGVGhR5oUK',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/77UjLW8j5UAGAGVGhR5oUK', 'duration_ms': 211440,
#      'time_signature': 4},
#     {'danceability': 0.763, 'energy': 0.602, 'key': 0, 'loudness': -9.696, 'mode': 1, 'speechiness': 0.0463,
#      'acousticness': 0.0284, 'instrumentalness': 0.151, 'liveness': 0.137, 'valence': 0.132, 'tempo': 120.014,
#      'type': 'audio_features', 'id': '7J7cNv9fl9nbofKMpPng4J', 'uri': 'spotify:track:7J7cNv9fl9nbofKMpPng4J',
#      'track_href': 'https://api.spotify.com/v1/tracks/7J7cNv9fl9nbofKMpPng4J',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7J7cNv9fl9nbofKMpPng4J', 'duration_ms': 325497,
#      'time_signature': 4},
#     {'danceability': 0.694, 'energy': 0.77, 'key': 6, 'loudness': -5.335, 'mode': 1, 'speechiness': 0.149,
#      'acousticness': 0.176, 'instrumentalness': 1.1e-05, 'liveness': 0.118, 'valence': 0.163, 'tempo': 125.905,
#      'type': 'audio_features', 'id': '0E9ZjEAyAwOXZ7wJC0PD33', 'uri': 'spotify:track:0E9ZjEAyAwOXZ7wJC0PD33',
#      'track_href': 'https://api.spotify.com/v1/tracks/0E9ZjEAyAwOXZ7wJC0PD33',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0E9ZjEAyAwOXZ7wJC0PD33', 'duration_ms': 184560,
#      'time_signature': 4},
#     {'danceability': 0.366, 'energy': 0.802, 'key': 5, 'loudness': -6.695, 'mode': 1, 'speechiness': 0.0357,
#      'acousticness': 0.0228, 'instrumentalness': 0.644, 'liveness': 0.248, 'valence': 0.23, 'tempo': 123.889,
#      'type': 'audio_features', 'id': '1QKR9U1OnjOHXLKAEzRcN8', 'uri': 'spotify:track:1QKR9U1OnjOHXLKAEzRcN8',
#      'track_href': 'https://api.spotify.com/v1/tracks/1QKR9U1OnjOHXLKAEzRcN8',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1QKR9U1OnjOHXLKAEzRcN8', 'duration_ms': 230319,
#      'time_signature': 4},
#     {'danceability': 0.489, 'energy': 0.505, 'key': 10, 'loudness': -8.022, 'mode': 0, 'speechiness': 0.117,
#      'acousticness': 0.579, 'instrumentalness': 0.000333, 'liveness': 0.104, 'valence': 0.337, 'tempo': 163.255,
#      'type': 'audio_features', 'id': '0y1QJc3SJVPKJ1OvFmFqe6', 'uri': 'spotify:track:0y1QJc3SJVPKJ1OvFmFqe6',
#      'track_href': 'https://api.spotify.com/v1/tracks/0y1QJc3SJVPKJ1OvFmFqe6',
#      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0y1QJc3SJVPKJ1OvFmFqe6', 'duration_ms': 213707,
#      'time_signature': 4}]

# Selecting specific variables from audio_features_result to train model
# Variables considered for training => danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, id

variables = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
             'valence', 'tempo', 'id']

audio_features_dataset = list(map(lambda x: {
    'danceability': x['danceability'],
    'energy': x['energy'],
    'loudness': x['loudness'],
    'speechiness': x['speechiness'],
    'acousticness': x['acousticness'],
    'instrumentalness': x['instrumentalness'],
    'liveness': x['liveness'],
    'valence': x['valence'],
    'tempo': x['tempo'],
    # 'id': x['id']
}, audio_features_result), )

# Creating a dataframe from audio_features_result
dataframe = pd.DataFrame(audio_features_dataset)

# Adding target
dataframe['target'] = np.where(True, 1, 0)
print(dataframe)

# Split the dataframe into train, validation, and test
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Creating an input pipeline

batch_size = 32
train_ds = tu.df_to_dataset(train, batch_size=batch_size)
val_ds = tu.df_to_dataset(val, shuffle=True, batch_size=batch_size)
test_ds = tu.df_to_dataset(test, shuffle=True, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
               'valence', 'tempo']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = tu.get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Creating the model
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Visualize our connectivity graph
plot_model(model, show_shapes=True, rankdir="LR")

#Training the model
model.fit(train_ds, epochs=40, validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
