import pandas as pd
import numpy as np

import tensorflow as tf
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

def read_raw_text(filename):

    with open(filename, 'r', encoding = 'utf-8') as file:

        document = file.read()

    return document

def read_ann_file(PATH, filename): #filename e.g. 01_nut.a/xxaa.ann

    PATH = PATH

    document = read_raw_text(PATH + filename[:-4] + '.txt')
    df = pd.read_csv(PATH + filename, sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)



    token_df = df[df[1].str.contains('T')]

    list_tokens = []

    seek = 0

    for index, row in token_df.iterrows():

        text = re.findall('\t.*', row[2])[0][1:]
        entityLabel, start, end = re.findall('.*\t', row[2])[0][:-1].split(' ')
        start, end = int(start), int(end)

        if seek == start:
            res = [document[start:end], start, end, entityLabel]
            list_tokens.append(res)

        else:
#             print(seek, start)
            res = [document[seek:start], seek, start, 'O']
            list_tokens.append(res)

            res = [document[start:end], start, end, entityLabel]
            list_tokens.append(res)

        seek = end


    result_text = ''

    for t, start, end, ent in list_tokens:
        text = f'[{ent}]{t}[/{ent}]'
        result_text += text


    return result_text, list_tokens

def tokenize(text):

    nlp = spacy_thai.load()
    pattern = r'\[(.*?)\](.*?)\[\/(.*?)\]'
    tokenizer = RegexpTokenizer(pattern)


    text = re.sub(r'([ก-๏a-zA-Z\(\)\.\s0-9\-]*)(?=\[\w+\])', r'[O]\1[/O]', text)
    text = re.sub(r'([ก-๏a-zA-Z\(\)\.\s0-9\-]+)$', r'[O]\1[/O]', text)
    text = re.sub(r'\[O\](\s)*?\[\/O\]', '', text)
    t = tokenizer.tokenize(text)

    result = []
    text_list_ = []

    for i in t:

            if i[0] == i[2]:
                doc = pythainlp.syllable_tokenize(i[1])
                token_texts = []

                # doc = nlp('สวัสดีค้าบ ท่านผู้เจริญ')
                for token in doc:
                    token_texts.append(token)
#                     if token.whitespace_:  # filter out empty strings
#                         token_texts.append(token.whitespace_)


                if i[0] == 'O' :
                    for r in range(len(token_texts)):
                        result.append((token_texts[r],  i[0]))
                  # words.append(r)
                else:
                    for r in range(len(token_texts)):

                        if r == 0:
                            result.append((token_texts[r], 'B-' + i[0]))

                        else:
                            result.append((token_texts[r], 'I-' + i[0]))

    text_list_.append(result)

    words = []
    tags = []
    original_text = []
    poss = []
    contain_digit = []
    contain_punc = []
    contain_vowel = []

    thai_vowel = 'ะาิีุุึืโเแัำไใฤๅฦ'

    def check_condition(condition):

        if condition:
            return 'True'
        else:
            return 'False'

    for text in text_list_:
        w = []
        t = []
        o = ''
        p = []
        digit = []
        punc = []
        vowel = []
        for word in text:
            w.append(word[0])
            t.append(word[1])
    #         p.append(pythainlp.tag.pos_tag(word[0]))
            o += word[0]
            digit.append(check_condition(any(char.isdigit() for char in word[0])))
            punc.append(check_condition(any(p in word[0] for p in punctuation)))
            vowel.append(check_condition(any(p in word[0] for p in thai_vowel)))

        words.append(w)
        tags.append(t)
        contain_digit.append(digit)
        contain_punc.append(punc)
        contain_vowel.append(vowel)
    #     poss.append(p)
        original_text.append(o)


#     dff = pd.DataFrame({'original_text' : original_text,
#                         'words' : words,
#     #                     'pos' : poss,
#                         'contain_digit' : contain_digit,
#                         'contain_punc' : contain_punc,
#                         'contain_vowel' : contain_vowel,
#                         'tags' : tags})



    return words, tags, original_text, contain_digit, contain_punc, contain_vowel

def read_all_file(PATH):

    PATH = PATH
    assignee_folder_list = os.listdir(PATH)[3:3+15]

    result = {'original_text' : [],
              'words' : [],
              'tags' : [],
              'contain_digit' : [],
              'contain_punc' : [],
              'contain_vowel' : []}
    for assignee_folder in assignee_folder_list:
        text_folder_list = sorted(os.listdir(PATH + assignee_folder))
        text_folder_list = [i for i in text_folder_list if i[-3:] in ['ann', 'txt']]
        text_folder_list = set(map(lambda x : x[:-4], text_folder_list))


        for text_folder in text_folder_list:

            filename = assignee_folder + '/' + text_folder + '.ann'

            try:
                text, list_tokens = read_ann_file(PATH, filename)
                words, tags, original_text, contain_digit, contain_punc, contain_vowel = tokenize(text)
                result['original_text'].append(original_text)
                result['words'].append(words)
                result['tags'].append(tags)
                result['contain_digit'].append(contain_digit)
                result['contain_punc'].append(contain_punc)
                result['contain_vowel'].append(contain_vowel)
            except:
                print(filename)

    df = pd.DataFrame(result)

    return df

def return_train_test(df):

    df['pos'] = df['words'].apply(lambda x : [i[1] for i in pythainlp.tag.pos_tag(x)])

    max_len = max(df['words'].apply(lambda x: len(x)))

    train, test = train_test_split(df, random_state = 42, test_size = 0.2)

    word_set = sorted(set([i for sentence in train['words'] for i in sentence]))
    pos_set = sorted(set([i for pos in train['pos'] for i in pos]))
    tag_set = sorted(set([i for tag in train['tags'] for i in tag]))

    word2idx = dict([(v, k) for k, v in enumerate(word_set)])
    pos2idx = dict([(v, k) for k, v in enumerate(pos_set)])
    tag2idx = dict([(v, k) for k, v in enumerate(tag_set)])
    digit2idx = {'True' : 1, 'False' : 0, '<PAD>' : 2}
    punc2idx = {'True' : 1, 'False' : 0, '<PAD>' : 2}
    vowel2idx = {'True' : 1, 'False' : 0, '<PAD>' : 2}

    word2idx['<UNK>'] = len(word2idx)
    word2idx['<PAD>'] = len(word2idx)
    pos2idx['<UNK>'] = len(pos2idx)
    pos2idx['<PAD>'] = len(pos2idx)
    tag2idx['<PAD>'] = len(tag2idx)

    train['words_idx'] = train['words'].apply(lambda x: [word2idx[i] for i in x])
    train['pos_idx'] = train['pos'].apply(lambda x: [pos2idx[i] for i in x])
    train['tags_idx'] = train['tags'].apply(lambda x: [tag2idx[i] for i in x])
    train['contain_digit_idx'] = train['contain_digit'].apply(lambda x: [digit2idx[i] for i in x])
    train['contain_punc_idx'] = train['contain_punc'].apply(lambda x: [punc2idx[i] for i in x])
    train['contain_vowel_idx'] = train['contain_vowel'].apply(lambda x: [vowel2idx[i] for i in x])

    test_sent = []
    test_pos = []
    test_tag = []

    for sent in test['words']:
        t = []
        for i in sent:
            try:
                t.append(word2idx[i])
            except:
                t.append(word2idx['<UNK>'])

        test_sent.append(t)

    for sent in test['pos']:
        t = []
        for i in sent:
            try:
                t.append(pos2idx[i])
            except:
                t.append(pos2idx['<UNK>'])

        test_pos.append(t)

    for sent in test['tags']:
        t = []
        for i in sent:

            t.append(tag2idx[i])


        test_tag.append(t)

    test['words_idx'] = test_sent
    test['pos_idx'] = test_pos
    test['tags_idx'] = test_tag
    test['contain_digit_idx'] = test['contain_digit'].apply(lambda x: [digit2idx[i] for i in x])
    test['contain_punc_idx'] = test['contain_punc'].apply(lambda x: [punc2idx[i] for i in x])
    test['contain_vowel_idx'] = test['contain_vowel'].apply(lambda x: [vowel2idx[i] for i in x])

    mapping = {'tok2idx' : word2idx,
               'pos2idx' : pos2idx,
               'tag2idx' : tag2idx}

    return train, test, mapping, max_len