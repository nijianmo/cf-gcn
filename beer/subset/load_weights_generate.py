'''Example script to generate text from beer reviews
   Only using reviews under unsupervised learning
   Alternatively training both rating and reviews

cncbofh 6 5.0
12oz short brown bottle.		It poured rusty brown with a nice big head quickly receding to a thin lace that had problems clinging to the glass.		The smell reminded me of a heath bar. Lots of toffee and coffee in my sniffer.		The taste is fantastic. Full of biscuit malt and roasted chocolate flavors. Very good. The beer was light in the mouth and went down nice and easy. 		Will finish the six pack and will look for more. Good stuff.
bataround 6 3.0
This beer poured a ruby brown color and was topped off with a very creamy. Good carbonation. There's some roastiness and caramel notes in the aroma, but overall weak in that regard. Is this a porter or a brown ale? I'm not sure I could distinguish, but it doesn't matter because the flavor is very pleasant. Roasted chocolate, toffee, burnt coffee come through with a nice underlying bitterness. Very drinkable and well made.

'''
from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import LSTM, Dropout, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate, Add, Dot, concatenate, add, dot
from keras.optimizers import RMSprop, Adam, Adagrad, SGD
from keras.utils.data_utils import get_file
from keras.callbacks import History
from keras.regularizers import l1, l2
from keras import initializers

from keras import backend as K
import theano
import theano.tensor as T
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import random
import sys
import cPickle as pickle
import time

# load encoding
with open('data_dir/beer/encodings.pkl', 'rb') as fp:
    num_users,num_items,user_indices,indices_user,item_indices,indices_item,chars,char_indices,indices_char = pickle.load(fp)

START_TOKEN = '<STR>'
STOP_TOKEN = '<EOS>'

with open('data_dir/beer/setByReviewCount.pkl', 'rb') as fp:
   reviewDict = pickle.load(fp) 

# build model
mf_dim = 8
reg_mf = 0.001
reg_kernel = 0.001
latent_dim = mf_dim
regs=[0,0]
maxlen = 200
enable_dropout=True

# Embedding layers
MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                              embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[0]))
MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                              embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[1]))

Embedding_User_Bias = Embedding(num_users, 1, embeddings_initializer = initializers.random_normal(stddev=0.01),
                                embeddings_regularizer = l2(regs[0]), name = 'embedding_user_bias')
Embedding_Item_Bias = Embedding(num_items, 1, embeddings_initializer = initializers.random_normal(stddev=0.01),
                                embeddings_regularizer = l2(regs[1]), name = 'embedding_item_bias')


# build lstm model
seq_input = Input((maxlen, len(chars)), name = 'seq_input')
seq_user_input = Input((maxlen, ), name = 'seq_user_input')
seq_item_input = Input((maxlen, ), name = 'seq_item_input')

seq_user_latent_MF = MF_Embedding_User(seq_user_input)
seq_item_latent_MF = MF_Embedding_Item(seq_item_input)

seq_user_bias = Embedding_User_Bias(seq_user_input)
seq_item_bias = Embedding_Item_Bias(seq_item_input)

merged = concatenate([seq_user_latent_MF, seq_item_latent_MF, seq_user_bias, seq_item_bias])

seq_predict_vector = merged
#transform_dim = 256
#seq_predict_vector = Dense(transform_dim, activation='sigmoid', name = "latent_transform")(merged)
#seq_enable_dropout = True
#if seq_enable_dropout:
#    seq_predict_vector = Dropout(0.2)(seq_predict_vector)

seq_predict_vector = concatenate([seq_input, seq_predict_vector])

# LSTM layers
char_input = LSTM(256, input_shape=(maxlen, len(chars) + 2 + 2 * mf_dim), return_sequences=True, name = 'jointchar_input')(seq_predict_vector)
#char_input = LSTM(256, input_shape=(maxlen, len(chars) + transform_dim), return_sequences=True, name = 'jointchar_input')(seq_predict_vector)
char_layer1 = LSTM(256, return_sequences=True, name = 'char_layer1')(char_input)
dropout = Dropout(0.2, name = 'char_drop')(char_layer1)
charout = TimeDistributed(Dense(len(chars), name = 'char_timedense'))(dropout)

# prediction layer
seq_prediction = Activation('softmax', name = 'char_softmax')(charout)

# model
model = Model(inputs=[seq_input, seq_user_input, seq_item_input], outputs=seq_prediction)

lr = 0.01
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))

# load pretrain model
model.load_weights('save_dir/Pretrain/beer_pretrain_word2vec_GMFLogistic_seq_8_neg_4_hr_0.6205_ndcg_0.3819.h5', by_name=True)


# sample
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) ### why softmax again?!
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# count
counts = []
for count in reviewDict.keys():
    #if count >= 200:
    if count == 0:
        counts.append(count)

reviewList = []
for target_count in counts:
    userList = reviewDict[target_count].keys()
    for user in userList:
        reviewList.extend(reviewDict[target_count][user])

maxlen = 200
#for review in reviewList[:50]:
for review in reviewList[50:100]:
  
    print("**********") 
    print("user: %s" % str(review.user))
    print("item: %s" % str(review.beer.name))
    print("rating: %s" % str(review.rating_overall))

    user = user_indices[review.user]
    item = item_indices[review.beer.id]
    users = []
    users.append(user)

    # real review
    print('----- real review:')
    print(START_TOKEN + review.text)

    ### predict seq start from random position
    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [0.5, 1.0]:
        print('----- diversity:', diversity)
        seq = []
        generated = ''
        seq.append(START_TOKEN)
        generated += START_TOKEN
        sys.stdout.write(generated)
        x = np.zeros((1, maxlen, len(chars)))
        y = np.zeros((1, ), dtype=int)
        x_user = np.ones((1, maxlen)) * user
        x_item = np.ones((1, maxlen)) * item

        for i in xrange(2000): ### generate 1000 characters
            curr_char = seq[i]
            x[0, :-1, :] = x[0, 1:, :]
            x[0, -1, :] = 0
            x[0, -1, char_indices[curr_char]] = 1
            preds = model.predict([x, x_user, x_item],verbose=0)[0]
            next_index = sample(preds[maxlen-1], diversity)
            next_char = indices_char[next_index]

            if next_char == STOP_TOKEN:
                break
            seq.append(next_char)
            generated += next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print('\n') 
