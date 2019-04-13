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

from core import metrics
from core import keras_models_further, embeddings
from graph import io
import random
from keras.models import model_from_json


def generate_arrays_from_file1(data_list, batch_size):
    while 1:
        cnt = 0
        X = []
        X1 = []
        Y = []
        Y1 = []
        # random.shuffle(index_list)
        for i in range(data_list[0].shape[0]):
            X.append(data_list[0][i])
            X1.append(data_list[1][i])
            Y.append(data_list[0][i])
            Y1.append(data_list[1][i])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield ([np.array(X), np.array(X1), np.array(Y), np.array(Y1)])
                X = []
                X1 = []
                Y = []
                Y1 = []


def evaluate(model, data_input, gold_output):
    predictions = model.predict_generator(generate_arrays_from_file1(data_input, batch_size=1024), steps=int(len(gold_output)/1024), verbose=1)
    gold_output = gold_output[:int(len(gold_output)/1024)*1024]
    if len(predictions.shape) == 3:
        predictions_classes = np.argmax(predictions, axis=2)
        train_batch_f1 = metrics.accuracy_per_sentence(predictions_classes, gold_output)
        print("Results (per sentence): ", train_batch_f1)
        train_y_properties_stream = gold_output.reshape(gold_output.shape[0] * gold_output.shape[1])
        predictions_classes = predictions_classes.reshape(predictions_classes.shape[0] * predictions_classes.shape[1])
        class_mask = train_y_properties_stream != 0
        train_y_properties_stream = train_y_properties_stream[class_mask]
        predictions_classes = predictions_classes[class_mask]
    else:
        predictions_classes = np.argmax(predictions, axis=1)
        train_y_properties_stream = gold_output

    accuracy = metrics.accuracy(np.array(predictions_classes), np.array(train_y_properties_stream))
    micro_scores = metrics.compute_micro_PRF(np.array(predictions_classes), np.array(train_y_properties_stream), empty_label=keras_models_further.p0_index)
    print("Results: Accuracy: ", accuracy)
    print("Results: Micro-Average F1: ", micro_scores)
    return predictions_classes, predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('mode', choices=['train', 'optimize', 'train-continue'])
    parser.add_argument('train_sentences')
    parser.add_argument('val_sentences')
    parser.add_argument('--models_folder', default="./trainedmodels/")

    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode

    embedding_matrix, word2idx = embeddings.load(keras_models_further.model_params['wordembeddings'])
    print("Loaded embeddings:", embedding_matrix.shape)

    training_data, _ = io.load_relation_graphs_from_file(args.train_sentences, load_vertices=True)
    val_data, _ = io.load_relation_graphs_from_file(args.val_sentences, load_vertices=True)

    print("Training data size: {}".format(len(training_data)))
    print("Validation data size: {}".format(len(val_data)))

    max_sent_len = keras_models_further.model_params['max_sent_len']
    print("Max sentence length set to: {}".format(max_sent_len))

    to_one_hot = np_utils.to_categorical
    graphs_to_indices = keras_models_further.to_indices

    train_as_indices = list(graphs_to_indices(training_data, word2idx))
    print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))
    training_data = None
    n_out = len(keras_models_further.property2idx)
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
            #random.shuffle(index_list)
            for i in range(data_list[0].shape[0]):
                X.append(data_list[0][i])
                X1.append(data_list[1][i])
                Y.append(data_list[0][i])
                Y1.append(data_list[1][i])
                Z.append(index_list[i])
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
        model = getattr(keras_models_further, model_name)(keras_models_further.model_params, embedding_matrix, max_sent_len, n_out)
        if "continue" in mode:
            print("Load pre-trained weights")
            model.load_weights(args.models_folder + model_name + ".kerasmodel", by_name=True)
            # json_string = model.to_json()
            # model = model_from_json(json_string)

        train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
        val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

        callback_history = model.fit_generator(generate_arrays_from_file(train_as_indices, train_y_properties_one_hot, batch_size=1024),
                                               epochs=500, steps_per_epoch=int(len(train_as_indices[-1])/1024), verbose=1,
                                               validation_data=generate_arrays_from_file(val_as_indices, val_y_properties_one_hot, batch_size=1024),
                                               validation_steps=int(len(val_as_indices[-1])/1024),
                                               callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=30, verbose=1),
                                                          callbacks.ModelCheckpoint(args.models_folder +
                                                                                    model_name + ".kerasmodel",
                                                                                    monitor='val_loss', verbose=1,
                                                                                    save_best_only=True)])

    print("Loading the best model")
    model = getattr(keras_models_further, model_name)(keras_models_further.model_params, embedding_matrix, max_sent_len, n_out)
    model.load_weights(args.models_folder + model_name + ".kerasmodel")


    print("Results on the training set")
    evaluate(model, train_as_indices[:-1], train_as_indices[-1])
    print("Results on the validation set")
    evaluate(model, val_as_indices[:-1], val_as_indices[-1])
