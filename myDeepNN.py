# -*- coding: utf-8 -*-
"""
Created on Thu May 24 00:27:33 2018

@author: mwahdan
"""

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from metrics import pearson_correlation
import numpy as np
import argparse
import time
from scipy import stats
from sklearn.metrics import mean_squared_error
from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import Vectorizer, VectorizerPosTags
from data_reader import read_SICK_data, read_dataset
from utils import pad_tensor, str2bool
from siamese import SiameseModel
import tensorflow as tf
from keras.optimizers import Adam

def createWordModel():
    n_hidden = 50
    input_dim = 50

    # unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.
    # he_normal: Gaussian initialization scaled by fan_in (He et al., 2014)

    lstm = layers.LSTM(n_hidden, unit_forget_bias=True,
                       kernel_initializer='he_normal',
                       kernel_regularizer='l2',
                       name='lstm1_layer')

    # Building the left branch of the model: inputs are variable-length sequences of vectors of size 128.
    left_input = Input(shape=(None, input_dim), name='input_1')
    #        left_masked_input = layers.Masking(mask_value=0)(left_input)
    left_output = lstm(left_input)

    # Building the right branch of the model: when you call an existing layer instance, you reuse its weights.
    right_input = Input(shape=(None, input_dim), name='input_2')
    #        right_masked_input = layers.Masking(mask_value=0)(right_input)
    right_output = lstm(right_input)

    # Instantiating and training the model: when you train such a model, the weights of the LSTM layer are updated based on both inputs.
    model = Model([left_input, right_input], [left_output, right_output])
    return model


def createPOSModel():
    n_hidden = 50
    input_dim = 50

    # unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.
    # he_normal: Gaussian initialization scaled by fan_in (He et al., 2014)

    lstm = layers.LSTM(n_hidden, unit_forget_bias=True,
                       kernel_initializer='he_normal',
                       kernel_regularizer='l2',
                       name='lstm2_layer')

    # Building the left branch of the model: inputs are variable-length sequences of vectors of size 128.
    left_input = Input(shape=(None, input_dim), name='input_3')
    #        left_masked_input = layers.Masking(mask_value=0)(left_input)
    left_output = lstm(left_input)

    # Building the right branch of the model: when you call an existing layer instance, you reuse its weights.
    right_input = Input(shape=(None, input_dim), name='input_4')
    #        right_masked_input = layers.Masking(mask_value=0)(right_input)
    right_output = lstm(right_input)

    # Instantiating and training the model: when you train such a model, the weights of the LSTM layer are updated based on both inputs.
    model = Model([left_input, right_input], [left_output, right_output])
    return model

def loadWordModel(filePath):
    print("Loading Word Model")
    # "E:\dataset\glove\glove.6B.50d.txt"
    f = open(filePath, 'r', encoding="utf8")
    model = dict()
    for line in f:
        splitLine = line.strip().replace(" ", "\t").split('\t')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


if __name__=='__main__':
    # read parameters
    train_df, dev_df = read_dataset("E:\\dataset\\sick2014\\SICK_train.csv", split=True, normalize_scores=True)
    test_df = read_dataset("E:\\dataset\\sick2014\\SICK_trial.csv", split=False, normalize_scores=True)
    pretrained = None
    save_path = "output.csv"
    # initialize objects
    print('Initializing objects ...')
    print('Initializing word embeddings ...')
    t1 = time.time()
    # /media/reza/book/dataset/word2vec/GoogleNews-vectors-negative300.bin
    # word_embeddings = WordEmbeddings("/media/reza/book/dataset/word2vec/GoogleNews-vectors-negative300.bin")
    word_embeddings = loadWordModel("E:\\dataset\\glove\\glove.6B.50d.txt")
    # /media/reza/book/Py_Projects/Lample2016-tagger-master/model_tag2vec.txt
    pos_embeddings = loadWordModel("E:\\Py_Projects\\Lample2016-tagger-master\\model_tag2vec.txt")

    t2 = time.time()
    print('\tTook %f seconds' % (t2 - t1))
    print('Initializing tokenizer ...')
    tokenizer = Tokenizer()
    print('Initializing vectorizer ...')
    vectorizer = Vectorizer(word_embeddings, tokenizer)
    vectorizer_pos = VectorizerPosTags(pos_embeddings)

    #### training dataset ####
    # vectorizing
    ids, train_a_vectors, train_b_vectors, train_gold = vectorizer.vectorize_df(train_df)
    train_a_pos_vectors, train_b_pos_vectors = vectorizer_pos.vectorize_sentence_pos_df(train_df)

    train_max_a_length = len(max(train_a_vectors, key=len))
    train_max_b_length = len(max(train_b_vectors, key=len))
    print('maximum number of tokens per sentence A in training set is %d' % train_max_a_length)
    print('maximum number of tokens per sentence B in training set is %d' % train_max_b_length)
    max_len = max([train_max_a_length, train_max_b_length])

    # padding
    train_a_vectors = pad_tensor(train_a_vectors, max_len)
    train_b_vectors = pad_tensor(train_b_vectors, max_len)

    #### development dataset ####
    # vectorizing
    ids, dev_a_vectors, dev_b_vectors, dev_gold = vectorizer.vectorize_df(dev_df)
    dev_a_pos_vectors, dev_b_pos_vectors = vectorizer_pos.vectorize_sentence_pos_df(dev_df)

    dev_max_a_length = len(max(dev_a_vectors, key=len))
    dev_max_b_length = len(max(dev_b_vectors, key=len))
    print('maximum number of tokens per sentence A in dev set is %d' % dev_max_a_length)
    print('maximum number of tokens per sentence B in dev set is %d' % dev_max_b_length)
    max_len = max([dev_max_a_length, dev_max_b_length])

    # padding
    dev_a_vectors = pad_tensor(dev_a_vectors, max_len)
    dev_b_vectors = pad_tensor(dev_b_vectors, max_len)

    print('Training the model ...')
    # model definition
    model11 = createWordModel()
    model12 = createPOSModel()
    sen1_output = K.concatenate([model11.output[0], model12.output[0]])
    sen2_output = K.concatenate([model11.output[1], model12.output[1]])
    #breakpoint()

    l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
    merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0],
                           name='L1_distance')([sen1_output, sen2_output])
    predictions = layers.Dense(1, activation='sigmoid', name='Similarity_layer')(merged)
    siamese = Model([model11.input, model12.input], predictions)
    # temp = K.concatenate([model11.input[0], model11.input[1], model12.input[0], model12.input[1]])
    # Compile the model
    optimizer = Adadelta()  # gradient clipping is not there in Adadelta implementation in keras
    #        optimizer = 'adam'
    siamese.compile(loss='mse', optimizer=optimizer, metrics=[pearson_correlation])

    # Train the model
    #if pretrained is not None:
    #    siamese.load_pretrained_weights(model_wieghts_path=pretrained)
    validation_data = ([[dev_a_vectors, dev_b_vectors], [dev_a_pos_vectors, dev_b_pos_vectors]], dev_gold)
    t1 = time.time()
    #breakpoint()

    history = siamese.fit([[train_a_vectors, train_b_vectors], [train_a_pos_vectors, train_b_pos_vectors]],
                train_gold)
    # **{'epochs':10, 'batch_size' :128}
    breakpoint()
    visualize_metric(history.history, 'loss')
    visualize_metric(history.history, 'pearson_correlation')
    load_activation_model()

    t2 = time.time()
    print('Took %f seconds' % (t2 - t1))
    if save_path is not None:
        model.save(model_folder=save_path)

    #### testing dataset ####
    print('Vectorizing testing dataset ...')
    ids, test_a_vectors, test_b_vectors, test_gold = vectorizer.vectorize_df(test_df)
    test_max_a_length = len(max(test_a_vectors, key=len))
    test_max_b_length = len(max(test_b_vectors, key=len))
    print('maximum number of tokens per sentence A in testing set is %d' % test_max_a_length)
    print('maximum number of tokens per sentence B in testing set is %d' % test_max_b_length)
    max_len = max([test_max_a_length, test_max_b_length])

    # padding
    print('Padding testing dataset ...')
    test_a_vectors = pad_tensor(test_a_vectors, max_len)
    test_b_vectors = pad_tensor(test_b_vectors, max_len)

    print('Testing the model ...')
    # Don't rely on evaluate method
    # result = siamese.evaluate(test_a_vectors, test_b_vectors, test_gold, 4906)
    # print(result)

    y = siamese.predict(test_a_vectors, test_b_vectors)
    y = [i[0] for i in y]
    assert len(test_gold) == len(y)

    mse = mean_squared_error(test_gold, y)
    print('MSE = %.2f' % mse)

    pearsonr = stats.pearsonr(test_gold, y)
    print('Pearson correlation (r) = %.2f' % pearsonr[0])

    spearmanr = stats.spearmanr(test_gold, y)
    print('Spearman’s p = %.2f' % spearmanr.correlation)