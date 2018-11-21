'''Example script to training the char-level language model
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import LSTM, Dropout, Embedding
from keras.layers import merge
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import History 
import numpy as np
import random
import sys
import cPickle as pickle

START_TOKEN = '<STR>'
STOP_TOKEN = '<EOS>'

# load encoding
with open('../data_dir/beer/encodings.pkl', 'rb') as fp:
    nUser,nItem,user_indices,indices_user,item_indices,indices_item,chars,char_indices,indices_char = pickle.load(fp)

# load review data
with open('../data_dir/beer/beer_implicit_train_review.pkl', 'rb') as fp:
    reviews = pickle.load(fp)

seqs = []
for review in reviews:
    seq = []
    seq.append(START_TOKEN)
    seq.extend(list(review.text))
    seq.append(STOP_TOKEN)
    seqs.append(seq)

text = [c for seq in seqs for c in seq]
print('beer_implicit_train_review corpus length:', len(text)) # text is list of all chars

# cut text into sequences, and generate corresponding x and y by shifting by one
maxlen = 200 # sequence length
step = 201 # sequence length + 1
sentences = []
next_sentences = []
for i in xrange(0, len(text) - len(text) % step, step):
    sentences.append(text[i: i + maxlen])
    next_sentences.append(text[i + 1: i + maxlen + 1])
nseq = len(sentences)
print('Number of sequences:', nseq)

print('Build sequences...')
X = np.zeros((len(sentences), maxlen), dtype=int)
y = np.zeros((len(sentences), maxlen), dtype=int)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
for i, sentence in enumerate(next_sentences):
    for t, char in enumerate(sentence):
        y[i, t] = char_indices[char]


def batch_generator(X, y, nbatch_size, n_batch):
    idx = 0
    while True:
        if idx >= n_batch:
            idx = 0
        X_train = np.zeros((nbatch_size, maxlen, len(chars)), dtype=np.bool)
        y_train = np.zeros((nbatch_size, maxlen, len(chars)), dtype=np.bool)
        
        for n in xrange(0, nbatch_size): ### for each sample seq in the min-batch
            i = n + idx * nbatch_size ### for each character in the sample seq
            for j in xrange(0, maxlen):
                char_x = X[i, j]
                char_y = y[i, j]
                X_train[n, j, char_x] = 1
                y_train[n, j, char_y] = 1

        idx += 1
        yield (X_train, y_train)

# build the model: a single LSTM
print('Build model...')

seq_input = Input((maxlen, len(chars)), name = 'seq_input')

# LSTM layers
char_input = LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True, name = 'char_input')(seq_input)
char_layer1 = LSTM(256, return_sequences=True, name = 'char_layer1')(char_input)
dropout = Dropout(0.2, name = 'char_drop')(char_layer1)
charout = TimeDistributed(Dense(len(chars), name = 'char_timedense'))(dropout)

# Final prediction layer
seq_predict = Activation('softmax', name = 'char_softmax')(charout)

model = Model(inputs=seq_input, outputs=seq_predict)

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print(model.summary())

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) ### why softmax again?!
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
n_iter = 20
nbatch_size = 256
n_batch = len(X) / nbatch_size
for iteration in xrange(1, n_iter):
    print()
    print('-' * 50)
    print('Iteration', iteration)
   
    hist = model.fit_generator(batch_generator(X, y, nbatch_size, n_batch),
                               n_batch, epochs=1)

    model.save('charnet.h5')
    model.save_weights("charnet_weights.h5")
    ### predict seq start from random position
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.5,1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += ''.join(sentence)
        print('----- Generating with seed: "' + generated + '"')
        sys.stdout.write(generated)

        for i in xrange(200): ### generate 200 characters
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            preds = model.predict(x,verbose=0)[0]
            next_index = sample(preds[maxlen-1], diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char) # sentence is a list

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

