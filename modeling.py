import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Reshape, Concatenate, Activation
from tensorflow.keras.utils import plot_model
from transformers import TFAutoModel
from tensorflow.keras import backend as K
from focal_loss import sparse_categorical_focal_loss
from transformers import AutoModel
from tensorflow.keras.layers import concatenate
from keras_contrib.layers import CRF

# import tensorflow_hub as hub
# import tensorflow_text as text
import pythainlp
import spacy_thai
from nltk.tokenize import RegexpTokenizer
import re
import string
import os
from string import punctuation

from preprocessing import *

df_1 = read_all_file(PATH = 'data/rel_data_ann_1/')
df_2 = read_all_file(PATH = 'data/rel_data_ann_2/')

for i in df_q.columns:
    df_1[i] = df_1[i].apply(lambda x: x[0])
    df_2[i] = df_2[i].apply(lambda x: x[0])

df_1['pos'] = df_1['words'].apply(lambda x : [i[1] for i in pythainlp.tag.pos_tag(x)])
df_2['pos'] = df_2['words'].apply(lambda x : [i[1] for i in pythainlp.tag.pos_tag(x)])

df_3 = pd.read_csv('data/rel_data_csv_1/').drop(columns = 'Unnamed: 0')
for i in ['words', 'contain_digit', 'contain_punc', 'contain_vowel', 'tags', 'pos']:
    df_Coraline[i] = df_Coraline[i].str.strip('[]').str.split(', ').apply(lambda x: [i[1:-1] for i in x])

df = pd.concat([df_1, df_2, df_3], ignore_index = True)

train, test, mapping, max_len = return_train_test(df)

train['padded_words_idx'] = list(pad_sequences(train['words_idx'], maxlen = max_len, padding = 'post', value = mapping['tok2idx']['<PAD>']))
train['padded_pos_idx'] = list(pad_sequences(train['pos_idx'], maxlen = max_len, padding = 'post', value = mapping['pos2idx']['<PAD>']))
train['padded_tags_idx'] = list(pad_sequences(train['tags_idx'], maxlen = max_len, padding = 'post', value = mapping['tag2idx']['<PAD>']))
train['padded_contain_digit_idx'] = list(pad_sequences(train['contain_digit_idx'], maxlen = max_len, padding = 'post', value = 2))
train['padded_contain_punc_idx'] = list(pad_sequences(train['contain_punc_idx'], maxlen = max_len, padding = 'post', value = 2))
train['padded_contain_vowel_idx'] = list(pad_sequences(train['contain_vowel_idx'], maxlen = max_len, padding = 'post', value = 2))

test['padded_words_idx'] = list(pad_sequences(test['words_idx'], maxlen = max_len, padding = 'post', value = mapping['tok2idx']['<PAD>']))
test['padded_pos_idx'] = list(pad_sequences(test['pos_idx'], maxlen = max_len, padding = 'post', value = mapping['pos2idx']['<PAD>']))
test['padded_tags_idx'] = list(pad_sequences(test['tags_idx'], maxlen = max_len, padding = 'post', value = mapping['tag2idx']['<PAD>']))
test['padded_contain_digit_idx'] = list(pad_sequences(test['contain_digit_idx'], maxlen = max_len, padding = 'post', value = 2))
test['padded_contain_punc_idx'] = list(pad_sequences(test['contain_punc_idx'], maxlen = max_len, padding = 'post', value = 2))
test['padded_contain_vowel_idx'] = list(pad_sequences(test['contain_vowel_idx'], maxlen = max_len, padding = 'post', value = 2))

def focal_loss(y_true, y_pred):

    class_weight = [10,10,10,15,15,
                    10,10,10,10,15,
                    10,10,10,15,15,
                    10,10,10,10,10,
                    1, 0.01
                    ]
    loss = sparse_categorical_focal_loss(y_true, y_pred, gamma=2, class_weight = class_weight)

    return loss

def get_model(input_dim_long, input_len_long, n_tags):

    # input_dim_long = 6032 + 1

    # input_len_long = len(train['padded_words_idx'].iloc[0])

    n_tags = len(label)
    output_dim = 8

    model_words = Input(shape = (input_len_long,))
    emb_words = Embedding(input_dim=input_dim_long, output_dim=output_dim)(model_words)
    # output_words = Reshape(target_shape=(output_dim, input_len_long))(emb_words)

    model_pos = Input(shape = (input_len_long,))
    emb_pos = Embedding(input_dim=input_dim_long, output_dim=output_dim)(model_pos)
    # output_pos = Reshape(target_shape=(output_dim, input_len_long))(emb_pos)
    model_digit = Input(shape = (input_len_long,))
    emb_digit = Embedding(input_dim=input_dim_long, output_dim=output_dim)(model_digit)

    model_punc = Input(shape = (input_len_long,))
    emb_punc = Embedding(input_dim=input_dim_long, output_dim=output_dim)(model_punc)

    model_vowel = Input(shape = (input_len_long,))
    emb_vowel = Embedding(input_dim=input_dim_long, output_dim=output_dim)(model_vowel)


    input_model = [model_words, model_pos, model_digit, model_punc, model_vowel]

    output_embeddings = [emb_words, emb_pos, emb_digit, emb_punc, emb_vowel]

    output_model = Concatenate()(output_embeddings)
    output_model = Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(output_model)
    output_model = TimeDistributed(Dense(n_tags, activation="softmax"))(output_model)

    model = Model(inputs = input_model, outputs = output_model)

    model.compile(loss= [focal_loss],
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-08),
                metrics=['accuracy'])

def train_model(X, y, model):
    loss = list()
    hist = model.fit(X, y, batch_size=64,  verbose=1, epochs=60, validation_split=0.2 )
    loss.append(hist.history['loss'][0])
    return model, loss

    X_tr_words = []
    for i in train['padded_words_idx']:
        X_tr_words.append(i)
    X_tr_words = np.array(X_tr_words)

    X_tr_pos = []
    for i in train['padded_pos_idx']:
        X_tr_pos.append(i)
    X_tr_pos = np.array(X_tr_pos)

    X_tr_digit = []
    for i in train['padded_contain_digit_idx']:
        X_tr_digit.append(i)
    X_tr_digit = np.array(X_tr_digit)

    X_tr_punc = []
    for i in train['padded_contain_punc_idx']:
        X_tr_punc.append(i)
    X_tr_punc = np.array(X_tr_punc)

    X_tr_vowel = []
    for i in train['padded_contain_vowel_idx']:
        X_tr_vowel.append(i)
    X_tr_vowel = np.array(X_tr_vowel)

    y_train = [i for i in train['padded_tags_idx']]
    y_train = np.array(y_train)

if __name__ == '__main__':

    model = train_model([X_tr_words, X_tr_pos, X_tr_digit, X_tr_punc, X_tr_vowel], y_train, model)

    X_te_words = []
    for i in test['padded_words_idx']:
        X_te_words.append(i)
    X_te_words = np.array(X_te_words)

    X_te_pos = []
    for i in test['padded_pos_idx']:
        X_te_pos.append(i)
    X_te_pos = np.array(X_te_pos)

    X_te_digit = []
    for i in test['padded_contain_digit_idx']:
        X_te_digit.append(i)
    X_te_digit = np.array(X_te_digit)

    X_te_punc = []
    for i in test['padded_contain_punc_idx']:
        X_te_punc.append(i)
    X_te_punc = np.array(X_te_punc)

    X_te_vowel = []
    for i in test['padded_contain_vowel_idx']:
        X_te_vowel.append(i)
    X_te_vowel = np.array(X_te_vowel)

    y_pred = model[0].predict([X_te_words, X_te_pos, X_te_digit, X_te_punc, X_te_vowel])
    y_pred = np.argmax(y_pred, axis = 2)
    y_test = []
    for i in test['padded_tags_idx']:
        y_test.append(i)
    y_test = np.array(y_test)




    print(classification_report(y_test.reshape(y_pred.shape[0]*y_pred.shape[1]),
                                y_pred.reshape(y_pred.shape[0]*y_pred.shape[1]),
                            target_names = label.keys())
        )