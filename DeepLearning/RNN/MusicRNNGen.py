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
#print("\nExample song: ")
#print(example_song)
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
#print("There are", len(vocab), "unique characters in the dataset")

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

# print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
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
#print("Batch function works correctly!")
#print(f"{x_batch}, {y_batch}")

"""
For each of these vectors, each index is processed at a single time step, so for the input
at time step 0, the model receives the index for the first character in the sequence,
and tried to predict the index of the next character. At the next timestep, it does the same
thing, but the RNN considers the information from the previous step which is its updated
state in addition to the current input. 
"""

x_batch, y_batch = get_batch(vectorized_songs, seq_length= 5, batch_size= 1)


"""
for i, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))
"""
    
"""
Lets now build the RNN model, and the use the trained model to generate a new song.
We train the RNN using batches of song snippets from the dataset. 

The model is based off the LSTM architecture, where we use a state vector to maintain information
about the temporal relationships between consecutive characters. The final output
of the LSTM is then fed into a fully connected linear nn.Linear layer where we'll output a softmax over each
character in the vocabulary, and then sample from this distribution to predict the next character

nn.Embedding: input layer, consisting of a trainable lookup table that maps the 
numbers of each character to a vector with embedding_dim dimensions
nn.LSTM: LSTM network, with size hidden_size
nn.Linear: output layer, with vocab_size outputs
"""

#Defining the RNN model

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        #define each network layer, layer 1 is the embedding layer to transform indices 
        #into dense vectors of a fixed embedding size
        # inputs are vectors of integer indices, the embedding layer maps each integer
        # to a real-valued vector of dimension embedding_dim
        # essentially, the embedding layer converts high-dimensional data into
        # a lower-dimensional space. This helps models to understand and work with complex
        # data more efficiently
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        '''TODO: Layer 2: LSTM with hidden_size `hidden_size`. note: number of layers defaults to 1.
         Use the nn.LSTM() module from pytorch.'''
        #LSTM: long short term memory is an RNN architecture designed to address vanishing gradient problem
        self.lstm = nn.LSTM(input_size= embedding_dim, hidden_size= self.hidden_size, batch_first= True) # TODO
        #really important to do, haha went on a full dimensional analysis of everything
        # do batch_first so that the LSTM knows which shape value is which in construction of RNN
        
        '''TODO: Layer 3: Linear (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.'''
        self.fc = nn.Linear(in_features= self.hidden_size, out_features= vocab_size) # TODO
    
    def init_hidden(self, batch_size, device):
        #initialize the hidden state and cell state with zeros:
        #this is needed as nn.LSTM expects an initial state at each forward pass
        #cell states are a vital part of LSTMs, as they act like long-term memory
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))
        #1: one layer, one direction
        #batch_size: number of sequences in the batch
        #self.hidden_state = number of features in hidden state
        
        #.to(device) ensures both h_0 ad c_0 are on the same device as the model/input
        
    def forward(self, x, state = None, return_state = False):
        x = self.embedding(x) #we pass a batch of token indices to convert to a dense vector
        #x.shape: (12, 100) -> (12, 100, embedding_dim)
        print(x.size())
        
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
            print(x.size(0))
        out, state = self.lstm(x, state)
        #Note (refer to PyTorch docstrings)
        #LSTM takes in as input: x of shape (batch_size, seq_length, embedding_dim)
        # also takes in the state (h_0, c_0)
        # it will output out: shape (batch_size, seq_length, hidden_size) -> sequence of outputs for all time steps
        # and state: (tuple): (h_n, c_n)-> final hidden and cell states for continuation
        
        
        out = self.fc(out)
        #we note here that self.fc is nn.Linear(hidden_size, vocab_size)
        # For each  LSTM output vector, it computes logits
        # logits_t = W h_t + b \in \mathbb{R}^{vocab_size}
        
        return out if not return_state else (out, state)
        

#Instantiate the model:
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 1024
batch_size = 8
#note the powers of 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checks if the GPU is available

model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)

#print summary of model
print(model)

#let us now test the RNN model:
x, y = get_batch(vectorized_songs, seq_length = 100, batch_size= 32)
x = x.to(device)
y = y.to(device)

pred = model(x) #input x (shape (12, 10)) is passed through the Embedding layer
# -> shape (12, 100, 256), then into the LSTM -> output shape (12, 100, 1024)
# -> then through linear layer -> pred.shape (12, 100, vocab_size)
# the thing to note here is batch_size = 8 is just a default hyperparameter and so the model doesn't
# care too much about batch size- it generalizes to any batch size. 
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

#lets get predictions from the untrained model (to train it we need to define a loss function)
"""
To get predictions from the model, we sample from the output distribution, defined by a softmax
over the character vocabulary. This gives us actual character indices, meaning we are
using a categorical distribution to sample over the example prediction. This gives
us a prediction of the next character specifically its index at each timestep. 

torch.multinomial samples over a categorical distribution to generate predictions
Note here that we sample from this probability distribution rather than just taking the
argmax which introduce stochasticity into the generation
"""

sampled_indices = torch.multinomial(torch.softmax(pred[0], dim = -1), num_samples = 1)
#note here that pred[0].shape = (seq_length, vocab_size), meaning that for 
# each of the seq_length time steps, we have a vector of logits over the vocabulary
# logits are essentially unnormalised output like z. 
# softmax then converts the logits into probabilities where each row sums to 1 to
#ensure a valid categorical distribution. 
# torch.multinomial now samples one index per timestep from the distribution,
# each row (each timestep), multinomial samples one index from the vocab according to categorical
# probabiltiies
#sampled_indices.shape = (seq_length, 1) where each row is now th epredicted character
# index at that time step. 
sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
#this squeezes out the last dimension, so now its a 1D array of character indices
#sampled_indices.shape = (seq_length,)
#cpu().numpy(), if we were on a GPU, this detaches the tensor and moves it to 
# CPU for further processing and converts it to a NumPy array. 


#lets now decode this:
print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


#Now, lets train the model and establish the loss function

"""
To train the model on the classification task, we can use a form of the crossentropy
loss (negative log likelihood loss). Specifically, we will use PyTorch's CrossEntropyLoss
as it combines the application of a log-softmax and negative log likelihood in a single
class and accepts integer targets for categorical classification tasks. We will want
to compute the loss using the true targets -- the labels -- and the predicted targets -- the
logits.
"""
#Define loss function:

cross_entropy = nn.CrossEntropyLoss() #instantiates the function

def compute_loss(labels, logits):
    """
    Inputs:
        labels: (batch_size, seq_length)
        logits: (batch_size, seq_length, vocab_size)
        
    Outputs:
        loss: scalar cross entropy loss over the batch and sequence length
    """
    batched_labels = labels.view(-1) #this compresses the shape of labels to (B* L,)
    #this allows us to treat each timestep across all batches as one prediction example
    
    #we want to batch th logits so that the shape of the logits is (B * L, V)
    batched_logits = logits.view(-1, vocab_size)
    
    #compute cross-entropy loss using batched next characters and predictions
    loss = cross_entropy(batched_logits, batched_labels) 
    return loss #shape: (B*L,)

#let us compute the loss of the predictions of the untrained model

#y.shape #(batch_size, sequence_length)
#pred.shape #(batch_size, seq_length, vocab_size)

example_batch_loss = compute_loss(y, pred)

print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
print(f"scalar_loss:      {example_batch_loss.mean().item()}")

### Hyperparameter setting and optimization ###

vocab_size = len(vocab)

# Model parameters:
params = dict(
  num_training_iterations = 3000,  # Increase this to train longer
  batch_size = 8,  # Experiment between 1 and 64
  seq_length = 100,  # Experiment between 50 and 500
  learning_rate = 5e-3,  # Experiment between 1e-5 and 1e-1
  embedding_dim = 256,
  hidden_size = 1024,  # Experiment between 1 and 2048
)

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

#now we can set up for experiment tracking via Comet. 

#create Comet experiment to track training run

def create_experiment():
    
    #end prior experiments
    if 'experiment' in locals():
        experiment.end()
        
    #initiate comet experiment for tracking
    experiment = comet_ml.Experiment(api_key = COMET_API_KEY,
                                     project_name = "MusicRNNGen")
    
    #log our hyperparameters to the experiment
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()
    
    return experiment

#define optimizer and training operation

"""
- Instantiate new model and optimizer
- Use loss.backward(), to perform backpropagation
- To update model's parameters based on the computed gradients, we will take a step
with the optimizer via optimizer.step()
"""

model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"])

#move model to GPU
model.to(device)

#instantiate an optimzer with its learning rate, lets try Adam to start (torch.optim)
optimizer = optim.Adam(model.parameters(), lr = params["learning_rate"])

def train_step(x, y):
    #set the model's mode to train
    
    model.train()
    
    #zero gradients for every step
    optimizer.zero_grad() #clears accumulated gradients from previous training step
    
    #forward pass
    y_hat = model(x)
    
    #compute loss
    loss = compute_loss(y, y_hat)
    
    #backward pass
    #complete gradient computation and update step
    #1. backpropgate the loss
    #2. update the model paramters via optimizer
    loss.backward() #when this is called, it accumulates gradients
    optimizer.step()
    
    return loss

##################
# Begin training!#
##################

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])

    # Convert numpy arrays to PyTorch tensors
    x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    # Take a train step
    loss = train_step(x_batch, y_batch)

    # Log the loss to the Comet interface
    experiment.log_metric("loss", loss.item(), step=iter)

    # Update the progress bar and visualize within notebook
    history.append(loss.item())
    plotter.plot(history)

    # Save model checkpoint
    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

# Save the final trained model
torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()

# To generate the music, we have to feed the model a seed to gt it started
"""
Once the seed is generated, we can then iteratively predict each successive character
using trained RNN. More specifically, recall that RNN outputs a softmax over possible 
successive characters. For inference, iteratively sample from these distributions
then use the samples to encode a generated song in ABC format. Then write the file
and then convert to audio

Lets look at the prediction procedure:
- Initialize a seed start string and the RNN state, and set the number of characters
we want to generate

- Use the start string and the RNN state to obtain the probability distribution over the
next predicted character

- Sample from multinomial distribution to calculate the index of the predicted chharacter
This predicted character is then used as the next input to the model. 

- At each time step, the updated RNN state is fed back into the model, so that it now
has more context in making the next prediction. After predicting the next character,
the updated RNN states are again fed back into the model, which is how it learns sequence 
dependencies in the data, as it gets more information from the previous predictions.
"""

#prediction of a generated song:

def generate_text(model, start_string, generation_length = 1000):
    #evaluation step (generate ABC text using learned RNN model)
    
    #convert strings to numbers (vectorise)
    input_idx = [char2idx[c] for c in start_string]
    input_idx = torch.tensor([input_idx], dtype= torch.long).to(device)
    
    #initialize the hidden state
    state = model.init_hidden(input_idx.size(0), device)
    #input_idx.size(0) tells us batch size
    
    text_generated = []
    tqdm._instances.clear()
    
    for i in tqdm(range(generation_length)):
        predictions, hidden_state = model(input_idx, state, return_state = True)
        #output: (1, seq_len, vocab_size)
        
        #remove batch dimension
        predictions = predictions.squeeze(0)
        #removes dimension 0, if and only if its size is 1
        #i.e x.shape == (1, T, V)
        #x.squeeze(0).shape == (T, V)
        # in generation, our batch_size here is 1 as we enerate one sequence at a time
        #torch.multinomial expected a 1D probability vector, i.e shape (vocab_size,)
        last_prediction = predictions[-1]
        
        #use multimonial distribution to sample over the probabilities
        input_idx = torch.multinomial(torch.softmax(last_prediction, dim = -1), num_samples= 1)
        
        #add predicted character to generated text
        predicted_char = idx2char[input_idx.item()]
        text_generated.append(predicted_char)
        
        input_idx = input_idx.unsqueeze(0)
        
    return (start_string + ''.join(text_generated))

generated_text = generate_text(model, "X")

#lets now play th generated music
generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
    #synnthesize the waveform from a song
    waveform = mdl.lab1.play_song(song)
    
    #if its a valid song, play it
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
        
        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
        
        #save song to comet interface
        experiment.log_asset(wav_file_path)
        
experiment.end()