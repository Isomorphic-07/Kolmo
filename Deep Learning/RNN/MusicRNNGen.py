import comet_ml

COMET_API_KEY = "YxN22Gyp6o5A88sIVQutQikAj"

import torch
import torch.nn as nn
import torch.optim as optim

import mitdeeplearning as mdl

import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
assert torch.cuda.is_available(), "Please enable GPU from runtime settings"
assert COMET_API_KEY != "", "Please insert your Comet API Key"

#Download dataset
songs = mdl.lab.load_training_data()

#Print one of the songs to inspect
example_song = songs[0]
print("\nExample song: ")
print(example_song)
# Convert the ABC notation to audio file and listen to it
mdl.lab1.play_song(example_song)

"""
One important thing to think about is that this notation of music does not simply 
contain information on the notes being played, but additionally there is meta 
information such as the song title, key, and tempo. How does the number of different 
characters that are present in the text file impact the complexity of the learning 
problem? This will become important soon, when we generate a numerical representation 
for the text data.
"""
# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

"""
We want to train an RNN model to learn patterns in ABC music, and then use this model to
generate a new piece of music based on this learned information. So given a character, or
sequence of chars, what is the most probable next character. We input a sequence of
chars to the model and train the model to predict the output (the following char
at each time step)
"""

#Vectorize the text via generating lookup tables: one that maps characters to numbers
# and a second that maps numbers back to characters. Recall, we have just identified the unique 
# characters present in the text

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')