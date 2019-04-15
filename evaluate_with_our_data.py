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

from evaluation import metrics
from core import keras_models_further, embeddings
from graph import io
import scipy.io as sio


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


def evaluate(model, data_input, gold_output, model_name):
    predictions = model.predict_generator(generate_arrays_from_file1(data_input, batch_size=20),
                                          steps=int(len(gold_output) / 20), verbose=1)
    sio.savemat(model_name+'all_prediction.mat', {'data': predictions})
    sio.savemat(model_name+'all_groundtruth.mat', {'data': gold_output})
    if len(predictions.shape) == 3:
        predictions_classes = np.argmax(predictions, axis=2)
        # train_batch_f1 = metrics.accuracy_per_sentence(predictions_classes, gold_output)
        # print("Results (per sentence): ", train_batch_f1)
        train_y_properties_stream = gold_output.reshape(gold_output.shape[0] * gold_output.shape[1])
        predictions_classes = predictions_classes.reshape(predictions_classes.shape[0] * predictions_classes.shape[1])
        class_mask = train_y_properties_stream != 0
        train_y_properties_stream = train_y_properties_stream[class_mask]
        predictions_classes = predictions_classes[class_mask]
        predictions = np.squeeze(predictions, axis=1)
    else:
        predictions_classes = np.argmax(predictions, axis=1)
        train_y_properties_stream = gold_output

    accuracy = metrics.accuracy(predictions_classes, train_y_properties_stream)
    print("Results: Accuracy: ", accuracy)
    print(predictions.shape)
    micro_curve = metrics.compute_precision_recall_curve(predictions, train_y_properties_stream, micro=True, empty_label = keras_models_further.p0_index)
    with open(model_name + "all_micro_curve.dat", 'w') as out:
        out.write("\n".join(["{}\t{}".format(*t) for t in micro_curve]))
    print("Micro precision-recall-curve stored in:", "./data/micro_curve.dat")
    macro_curve = metrics.compute_precision_recall_curve(predictions, train_y_properties_stream, micro=False, empty_label = keras_models_further.p0_index)
    with open(model_name + "all_macro_curve.dat", 'w') as out:
        out.write("\n".join(["{}\t{}".format(*t) for t in macro_curve]))
    print("Macro precision-recall-curve stored in:", "./data/macro_curve.dat")
    return predictions_classes, predictions, micro_curve, macro_curve


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('mode', choices=['train', 'optimize', 'train-continue'])
    parser.add_argument('train_set')
    parser.add_argument('val_set')
    parser.add_argument('--models_folder', default="./trainedmodels/")

    args = parser.parse_args()

    model_name = args.model_name
    mode = args.mode

    embedding_matrix, word2idx = embeddings.load(keras_models_further.model_params['wordembeddings'])
    print("Loaded embeddings:", embedding_matrix.shape)

    val_data, _ = io.load_relation_graphs_from_file(args.val_set, load_vertices=True)
    print("Validation data size: {}".format(len(val_data)))

    max_sent_len = keras_models_further.model_params['max_sent_len']
    print("Max sentence length set to: {}".format(max_sent_len))

    to_one_hot = np_utils.to_categorical
    graphs_to_indices = keras_models_further.to_indices
    if "Context" in model_name:
        to_one_hot = embeddings.timedistributed_to_one_hot
        graphs_to_indices = keras_models_further.to_indices_with_extracted_entities
    elif "CNN" in model_name:
        graphs_to_indices = keras_models_further.to_indices_with_relative_positions

    val_as_indices = list(graphs_to_indices(val_data, word2idx))
    val_data = None
    n_out = len(keras_models_further.property2idx)
    print("N_out:", n_out)


    print("Loading the best model")
    model = getattr(keras_models_further, model_name)(keras_models_further.model_params, embedding_matrix, max_sent_len, n_out)
    model.load_weights(args.models_folder + model_name + ".kerasmodel", by_name=True)


    # print("Results on the training set")
    # evaluate(model, train_as_indices[:-1], train_as_indices[-1])
    print("Results on the validation set")
    i1, i2, i3, i4 = evaluate(model, val_as_indices[:-1], val_as_indices[-1], model_name)
