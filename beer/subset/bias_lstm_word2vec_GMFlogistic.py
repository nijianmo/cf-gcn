'''
Created on March 10, 2017

@author: jianmo
'''
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate, Add, Dot, concatenate, add, dot, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import cPickle as pickle
import logging
logging.basicConfig(level=logging.DEBUG)
import GMFlogistic, word2vec_GMFlogistic


def get_model(num_users, num_items, num_chars, EMBEDDING_DIM, user_text_embedding_matrix, item_text_embedding_matrix, latent_dim, maxlen=200, regs=[0,0], reg_text = 0, enable_dropout=False):
    assert len(regs) == 2       # regularization for user and item, respectively
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[0]))
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[1]))   

    # load pre-calculated user text and item text embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    user_text_embedding_layer = Embedding(num_users,
                                          EMBEDDING_DIM,
                                          weights=[user_text_embedding_matrix],
                                          input_length=1,
                                          trainable=False)

    item_text_embedding_layer = Embedding(num_items,
                                          EMBEDDING_DIM,
                                          weights=[item_text_embedding_matrix],
                                          input_length=1,
                                          trainable=False)

    user_text_latent = Flatten()(user_text_embedding_layer(user_input))
    item_text_latent = Flatten()(item_text_embedding_layer(item_input))
    if enable_dropout:
        user_text_latent = Dropout(0.5)(user_text_latent)
        item_text_latent = Dropout(0.5)(item_text_latent)
   
    # Crucial to transform into lower latent dimenstion
    user_text_latent = Dense(latent_dim, kernel_regularizer=l2(reg_text), name = 'user_text_latent_transform')(user_text_latent)
    item_text_latent = Dense(latent_dim, kernel_regularizer=l2(reg_text), name = 'item_text_latent_transform')(item_text_latent)

    Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding_latent',
                               embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[0]), input_length=1)
    Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding_latent',
                               embeddings_initializer=initializers.random_normal(stddev=0.01), embeddings_regularizer = l2(regs[1]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_embedding_latent = Flatten()(Embedding_User(user_input))
    item_embedding_latent = Flatten()(Embedding_Item(item_input))

    # Element-wise product of user and item text embeddings 
    #predict_vector_text = multiply([user_text_latent, item_text_latent])
    predict_user_text = multiply([user_text_latent, item_embedding_latent])
    predict_item_text = multiply([item_text_latent, user_embedding_latent])
    predict_vector_text = concatenate([predict_user_text, predict_item_text])

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings 
    predict_vector = multiply([user_latent, item_latent])

    #predict_vector = add([predict_vector, predict_vector_text])
    predict_vector = concatenate([predict_vector, predict_vector_text])

    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction_pretrain')(predict_vector)

    # language model
    # sequence bias embedding
    Embedding_User_Bias = Embedding(num_users, 1, embeddings_initializer = initializers.random_normal(stddev=0.01),
                                    embeddings_regularizer = l2(regs[0]), name = 'embedding_user_bias')
    Embedding_Item_Bias = Embedding(num_items, 1, embeddings_initializer = initializers.random_normal(stddev=0.01),
                                    embeddings_regularizer = l2(regs[1]), name = 'embedding_item_bias')

    # Input variables
    seq_input = Input((maxlen, num_chars), name = 'seq_input')
    seq_user_input = Input((maxlen, ), name = 'seq_user_input')
    seq_item_input = Input((maxlen, ), name = 'seq_item_input')

    mf_seq_user_latent = MF_Embedding_User(seq_user_input)
    mf_seq_item_latent = MF_Embedding_Item(seq_item_input)

    seq_user_bias = Embedding_User_Bias(seq_user_input)
    seq_item_bias = Embedding_Item_Bias(seq_item_input)
    merged = concatenate([mf_seq_user_latent, mf_seq_item_latent, seq_user_bias, seq_item_bias])
    seq_predict_vector = merged
    #transform_dim = 256
    #seq_predict_vector = Dense(transform_dim, activation='sigmoid', name = "latent_transform")(merged)
    #seq_enable_dropout = True
    #if seq_enable_dropout:
    #    seq_predict_vector = Dropout(0.2)(seq_predict_vector)

    seq_predict_vector = concatenate([seq_input, seq_predict_vector])

    # LSTM layers
    char_input = LSTM(256, input_shape=(maxlen, len(chars) + 2 + 2 * latent_dim), return_sequences=True, name = 'jointchar_input')(seq_predict_vector)
    #char_input = LSTM(256, input_shape=(maxlen, len(chars) + transform_dim), return_sequences=True, name = 'jointchar_input')(seq_predict_vector)
    char_layer1 = LSTM(256, return_sequences=True, name = 'char_layer1')(char_input)
    dropout = Dropout(0.2, name = 'char_drop')(char_layer1)
    charout = TimeDistributed(Dense(len(chars), name = 'char_timedense'))(dropout)

    # Final prediction layer
    seq_prediction = Activation('softmax', name = 'char_softmax')(charout)

    eval_model = Model(inputs=[user_input, item_input], 
        outputs=prediction)

    seq_model = Model(inputs=[seq_input, seq_user_input, seq_item_input], 
        outputs=seq_prediction)    
    
    joint_model = Model(inputs=[seq_input, seq_user_input, seq_item_input, user_input, item_input], 
        outputs=[seq_prediction, prediction])

    return eval_model, seq_model, joint_model


def get_train_instances(train, num_negatives, weight_negatives, user_weights):
    user_input, item_input, labels, weights = [],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        weights.append(user_weights[u])
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)): # make sure there is no explicit pair (u,j)
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            weights.append(weight_negatives * user_weights[u])
    return user_input, item_input, labels, weights

def get_train_seqs(reviews, user_indices, item_indices, char_indices, maxlen = 200, START_TOKEN = '<STR>', STOP_TOKEN = '<EOS>'):
    # instances for review model
    # combine all reviews
    seqs = []
    text_users = []
    text_items = []
    for review in reviews:
        user = user_indices[review.user]
        item = item_indices[review.beer.id]
        #user = review[0]
        #item = review[1]
        seq = []
        seq.append(START_TOKEN)
        seq.extend(list(review.text))
        #seq.extend(list(review[2]))
        seq.append(STOP_TOKEN)
        seqs.append(seq)
        text_users.extend([user] * len(seq)) # Use extend
        text_items.extend([item] * len(seq))

    # text is list of all chars
    text = [c for seq in seqs for c in seq]
    logging.debug('beer_implicit-train corpus length: %u' % len(text))

    # cut text into sequences, and generate corresponding x and y by shifting by one
    maxlen = 200 # sequence length
    step = 201 # sequence length + 1

    sentences = []
    sentences_users = []
    sentences_items = []
    next_sentences = []
    for i in xrange(0, len(text) - len(text) % step, step):
        sentences.append(text[i: i + maxlen])
        sentences_users.append(text_users[i: i + maxlen])
        sentences_items.append(text_items[i: i + maxlen])
        next_sentences.append(text[i + 1: i + maxlen + 1])

    logging.debug('Number of sequences: %u' % len(sentences))
    logging.debug('Build sequences...')
    X = np.zeros((len(sentences), maxlen), dtype=int)
    y = np.zeros((len(sentences), maxlen), dtype=int)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t] = char_indices[char]
    for i, sentence in enumerate(next_sentences):
        for t, char in enumerate(sentence):
            y[i, t] = char_indices[char]

    X_user = np.zeros((len(sentences), maxlen), dtype=int)
    X_item = np.zeros((len(sentences), maxlen), dtype=int)
    for i, sentence in enumerate(sentences_users):
        for t, user in enumerate(sentence):
            X_user[i, t] = user

    for i, sentence in enumerate(sentences_items):
        for t, item in enumerate(sentence):
            X_item[i, t] = item

    return X, y, X_user, X_item

#seq_batch_idx = 0
batch_idx = 0
def batch_generator(X, X_user, X_item, y, user_input, item_input, labels, batch_size, num_batch, seq_num_batch):
   #global seq_batch_idx
   #batch_idx = 0 # each epoch run through all implicit interactions
   
   global batch_idx
   seq_batch_idx = 0 # each epoch run through all reviews

   while True:
        if batch_idx >= num_batch:
            batch_idx = 0
        if seq_batch_idx >= seq_num_batch:
            seq_batch_idx = 0

        X_train = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
        y_train = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
        X_user_train = X_user[seq_batch_idx * batch_size : seq_batch_idx * batch_size + batch_size]
        X_item_train = X_item[seq_batch_idx * batch_size : seq_batch_idx * batch_size + batch_size]

        for s in xrange(0, batch_size): # for each sample seq in the min-batch
            i = seq_batch_idx * batch_size + s # for each character in the sample seq
            for c in xrange(0, maxlen):
                charx = X[i, c]
                chary = y[i, c]
                X_train[s, c, charx] = 1
                y_train[s, c, chary] = 1
        seq_batch_idx += 1

        start = batch_idx * batch_size
        end = batch_idx * batch_size + batch_size

        user = np.array(user_input[start:end])
        item = np.array(item_input[start:end])
        label = np.array(labels[start:end])
        batch_idx += 1

        X_out = {'seq_input':X_train, 'seq_user_input':X_user_train, 'seq_item_input':X_item_train, 'user_input':user, 'item_input':item}
        yield (X_out, [y_train, label])

if __name__ == '__main__':
    dataset_name = "beer"
    num_factors = 8
    regs = [0,0]
    reg_text = 0.01
    num_negatives = 4
    weight_negatives = 1
    learner = "Adam"
    learning_rate = 0.001
    epochs = 100
    batch_size = 256
    verbose = 1
    word2vec_mf_pretrain = 'save_dir/Pretrain/beer_GMF_8_neg_4_hr_0.5865_ndcg_0.3345.h5'
    transform_dim = 256
    maxlen = 200 # sequence length
    enable_dropout = False

    if (len(sys.argv) > 3):
        dataset_name = sys.argv[1]
        num_factors = int(sys.argv[2])
        regs = eval(sys.argv[3])
        num_negatives = int(sys.argv[4])
        weight_negatives = float(sys.argv[5])
        learner = sys.argv[6]
        learning_rate = float(sys.argv[7])
        epochs = int(sys.argv[8])
        batch_size = int(sys.argv[9])   
        verbose = int(sys.argv[10])
        
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    logging.debug("pretrain_lstm_word2vec_GMF-logistic (%s) Settings: num_factors=%d, batch_size=%d, learning_rate=%.1e, num_neg=%d, weight_neg=%.2f, regs=%s, reg_text=%.3f, epochs=%d, verbose=%d"
          %(learner, num_factors, batch_size, learning_rate, num_negatives, weight_negatives, regs, reg_text, epochs, verbose))

    # load encoding
    with open('data_dir/beer/encodings.pkl', 'rb') as fp:
        num_users,num_items,user_indices,indices_user,item_indices,indices_item,chars,char_indices,indices_char = pickle.load(fp)
    num_chars = len(chars)

    # load review data
    with open('data_dir/beer/beer_implicit_train_review.pkl', 'rb') as fp:
        reviews = pickle.load(fp)
    
    # Loading data
    t1 = time()
    dataset = Dataset("save_dir/Data/" + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    total_weight_per_user = train.nnz / float(num_users)
    train_csr, user_weights = train.tocsr(), []
    for u in xrange(num_users):
        user_weights.append(1)
    logging.debug("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # load text data 
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 64

    logging.debug("MAX_SEQUENCE_LENGTH=%d, MAX_NB_WORDS=%d, EMBEDDING_DIM=%d" % (MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,EMBEDDING_DIM))

    with open('save_dir/useritem_text_embedding.pkl','rb') as fp:
        user_text_embedding_matrix, item_text_embedding_matrix = pickle.load(fp)

     # Build model
    eval_model, seq_model, joint_model = get_model(num_users, num_items, num_chars, 
    	EMBEDDING_DIM, user_text_embedding_matrix, item_text_embedding_matrix, 
        num_factors, maxlen, regs, reg_text, enable_dropout)

    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        eval_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        seq_model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy')
        joint_model.compile(optimizer=Adam(lr=learning_rate),
        	loss={'char_softmax': 'categorical_crossentropy', 'prediction_pretrain': 'binary_crossentropy'},
                loss_weights={'char_softmax': 1.0 / num_chars, 'prediction_pretrain': 1.0})
        	#loss_weights={'char_softmax': 0.9, 'prediction': 0.1})
        	#loss_weights={'char_softmax': 0.1, 'prediction': 0.9})
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    #logging.debug(model.summary())

    # Load pretrain model
    if word2vec_mf_pretrain != '':
        eval_model.load_weights(word2vec_mf_pretrain, by_name=True)
        print("Load pretrained word2vec_GMF (%s) model done. " %(word2vec_mf_pretrain))

    # Load pre-trained character rnn
    joint_model.load_weights('lm/charnet_weights.h5', by_name=True)
    charnet = load_model('lm/charnet.h5')
    char_input_weight = charnet.get_layer('char_input').get_weights()
    char_input_weight[0] = np.vstack([char_input_weight[0],
    	                              np.zeros((2 + 2 * num_factors, 256 * 4)).astype(
                                      # np.zeros((transform_dim, 256 * 4)).astype(
                                      np.float32)])
    joint_model.get_layer('jointchar_input').set_weights(char_input_weight)    


    # Init performance
    (hits, ndcgs) = evaluate_model(eval_model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    mf_embedding_norm = np.linalg.norm(eval_model.get_layer('user_embedding').get_weights())+np.linalg.norm(eval_model.get_layer('item_embedding').get_weights())
    p_norm = np.linalg.norm(eval_model.get_layer('prediction_pretrain').get_weights()[0])
    logging.debug('Init: HR = %.4f, NDCG = %.4f\t MF_norm=%.1f, p_norm=%.2f' % 
          (hr, ndcg, mf_embedding_norm, p_norm))

    if hr > 0.75:
        eval_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_eval_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
        seq_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_seq_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
        joint_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_joint_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
 
    # Generate training sequences
    X, y, X_user, X_item = get_train_seqs(reviews, user_indices, item_indices, char_indices, maxlen = 200, START_TOKEN = '<STR>', STOP_TOKEN = '<EOS>')
    batch_size = 256
    seq_num_batch = len(X) / batch_size
    
    # Train model
    loss_pre = sys.float_info.max
    best_hr, best_ndcg = 0, 0
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels, weights = get_train_instances(train, num_negatives, weight_negatives, user_weights)
        num_batch = len(user_input) / batch_size 

        # Training
        hist = joint_model.fit_generator(batch_generator(X, X_user, X_item, y, user_input, item_input, labels, batch_size, num_batch, seq_num_batch),
                                           seq_num_batch, epochs=1, verbose=1) # run through all reviews 
                                           #num_batch, epochs=1, verbose=1) # run through all interactions 
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(eval_model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            mf_embedding_norm = np.linalg.norm(eval_model.get_layer('user_embedding').get_weights())+np.linalg.norm(eval_model.get_layer('item_embedding').get_weights())
            p_norm = np.linalg.norm(eval_model.get_layer('prediction_pretrain').get_weights()[0])
            logging.debug('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s] MF_norm=%.1f, p_norm=%.2f' 
                  % (epoch,  t2 - t1, hr, ndcg, loss, time()-t2, mf_embedding_norm, p_norm))
            if hr > 0.5:
                eval_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_eval_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
                seq_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_seq_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)
                joint_model.save_weights('save_dir/Pretrain/%s_pretrain_word2vec_GMFLogistic_joint_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, num_factors, num_negatives, hr, ndcg), overwrite=True)		           
            if hr > best_hr:
                best_hr = hr
            if ndcg > best_ndcg:
                best_ndcg = ndcg

    logging.debug("End. best HR = %.4f, best NDCG = %.4f" %(best_hr, best_ndcg))
