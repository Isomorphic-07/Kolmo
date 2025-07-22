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
songs = mdl.lab1.load_training_data()

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
"""
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
"""
### Vectorize the songs string ###

'''TODO: Write a function to convert the all songs string to a vectorized
    (i.e., numeric) representation. Use the appropriate mapping
    above to convert from vocab characters to the corresponding indices.

  NOTE: the output of the `vectorize_string` function
  should be a np.array with `N` elements, where `N` is
  the number of characters in the input string
'''
def vectorize_string(string):
    '''TODO'''
    """
    n = len(string)
    vectorString = []
    for character in string:
        vectorString.append(char2idx[character])
    return np.array(vectorString)
    """
    return np.array([char2idx[char] for char in string])
  

vectorized_songs = vectorize_string(songs_joined)

print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"

#creating training examples and targets: We need to divide the text into example sequences
# to be used for training, where each input sequence that we feed into our RNN contains seq_length
# Additionally, we will also define a target sequence for input sequence, which will be used 
# in training the RNN to predict the next character. For each input, the corresponding
# target will contain the same length of text, except shifted one character to the right.

"""
To do this, we'll break the text into chuncks of seq_length + 1, suppose seq_length is 4
and our text is "hello". Then, our input is "hell" and the target sequence is "ello".
The batch method will let us convert this stream of character indices to sequences
of the desired size.
"""

#Batch definition to create training examples

def get_batch(vectorized_songs, seq_length, batch_size):
    # length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1 #it essentially tells us the number of valid character transitions in the dataset
    #think of it as the maximum starting point for your input target pairs like x[t] -> y[t+1]
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)
    # this returns a random sample of batch_size number of elements from range(n - seq_length)
    # essentially if we exceed n-seq_length, we will get overflow
    
    '''TODO: construct a list of input sequences for the training batch'''
    input_batch = np.array([vectorized_songs[n: n + seq_length] for n in idx])
    
    #essentially the target batch, we want to predict these
    '''TODO: construct a list of output sequences for the training batch'''
    output_batch = np.array([vectorized_songs[n + 1: n + seq_length + 1] for n in idx])
    
     # Convert the input and output batches to tensors
    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)

    return x_batch, y_batch

test_args = (vectorized_songs, 10, 2)
x_batch, y_batch = get_batch(*test_args)
assert x_batch.shape == (2, 10), "x_batch shape is incorrect"
assert y_batch.shape == (2, 10), "y_batch shape is incorrect"
print("Batch function works correctly!")
#print(f"{x_batch}, {y_batch}")

"""
For each of these vectors, each index is processed at a single time step, so for the input
at time step 0, the model receives the index for the first character in the sequence,
and tried to predict the index of the next character. At the next timestep, it does the same
thing, but the RNN considers the information from the previous step which is its updated
state in addition to the current input. 
"""

x_batch, y_batch = get_batch(vectorized_songs, seq_length= 5, batch_size= 1)

for i, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))