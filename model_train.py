# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import numpy as np
np.random.seed(1)

from keras import callbacks
from keras.utils import np_utils
import hyperopt as hy
import json

#from evaluation import metrics
from core import keras_models, embeddings
from graph import io
import random


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('mode', choices=['train', 'optimize', 'train-continue'])
    parser.add_argument('train_sentences')
    parser.add_argument('train_pairs')
    parser.add_argument('val_sentences')
    parser.add_argument('val_pairs')
    parser.add_argument('--models_folder', default="./trainedmodels/")

    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode

    embedding_matrix, word2idx = embeddings.load(keras_models.model_params['wordembeddings'])
    print("Loaded embeddings:", embedding_matrix.shape)

    training_data, _ = io.load_relation_graphs_from_file(args.train_sentences, load_vertices=True)
    with open(args.train_pairs) as f:
        train_pairs = json.load(f)
    val_data, _ = io.load_relation_graphs_from_file(args.val_sentences, load_vertices=True)
    with open(args.val_pairs) as f:
        val_pairs = json.load(f)

    print("Training data size: {}".format(len(training_data)))
    print("Validation data size: {}".format(len(val_data)))

    max_sent_len = keras_models.model_params['max_sent_len']
    print("Max sentence length set to: {}".format(max_sent_len))

    to_one_hot = np_utils.to_categorical
    graphs_to_indices = keras_models.to_indices

    train_as_indices = list(graphs_to_indices(training_data, word2idx))
    print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))
    training_data = None
    n_out = len(keras_models.property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx))
    val_data = None


    def generate_arrays_from_file(data_list, index_list, batch_size):
        while 1:
            cnt = 0
            X = []
            X1 = []
            Y = []
            Y1 = []
            Z = []
            random.shuffle(index_list)
            for pair in index_list:
                X.append(data_list[0][pair[0]])
                X1.append(data_list[1][pair[0]])
                Y.append(data_list[0][pair[1]])
                Y1.append(data_list[1][pair[1]])
                Z.append(pair[2])
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    yield ([np.array(X), np.array(X1), np.array(Y), np.array(Y1)], np.array(Z))
                    X = []
                    X1 = []
                    Y = []
                    Y1 = []
                    Z = []

    if "train" in mode:
        print("Training the model")
        print("Initialize the model")
        model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
        if "continue" in mode:
            print("Load pre-trained weights")
            model.load_weights(args.models_folder + model_name + ".kerasmodel")

        # train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
        # val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

        callback_history = model.fit_generator(generate_arrays_from_file(train_as_indices, train_pairs, batch_size=1024),
                                               epochs=50, steps_per_epoch=int(len(train_pairs)/1024), verbose=1,
                                               validation_data=generate_arrays_from_file(val_as_indices, val_pairs, batch_size=1024),
                                               validation_steps=int(len(val_pairs)/1024),
                                               callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                                                          callbacks.ModelCheckpoint(args.models_folder +
                                                                                    model_name + ".kerasmodel",
                                                                                    monitor='val_loss', verbose=1,
                                                                                    save_best_only=True)])
