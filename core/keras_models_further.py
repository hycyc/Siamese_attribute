# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import os
import ast, json
import numpy as np
import tensorflow as tf
np.random.seed(1)

from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers
import tqdm

from core import embeddings
from graph import graph_utils

import random
#from keras.models import Sequential, Model
#from keras.layers import Input, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "../model_params.json")) as f:
    model_params = json.load(f)

#property_blacklist = embeddings.load_blacklist(os.path.join(module_location, "..\\..\\resources\\property_blacklist.txt"))
property2idx = {}
with open(os.path.join(module_location, model_params["property2idx"])) as f:
    property2idx = ast.literal_eval(f.read())
idx2property = {v: k for k, v in property2idx.items()}

_, position2idx = embeddings.init_random(np.arange(-model_params['max_sent_len'], model_params['max_sent_len']),
                                         1, add_all_zeroes=True)

p0_index = 1

MAX_EDGES_PER_GRAPH = 1
POSITION_EMBEDDING_MODE = "mark-bi"
POSITION_VOCAB_SIZE = 5 if POSITION_EMBEDDING_MODE == "mark-bi" and not graph_utils.LEGACY_MODE else 4
# 基础模型


def create_base_network(input_shape, embedding_matrix, p):
    # Take sentence encoded as indices and convert it to embeddings
    sentence_input = layers.Input(shape=(input_shape,))
    word_embeddings = layers.Embedding(output_dim=embedding_matrix.shape[1], input_dim=embedding_matrix.shape[0],
                                       input_length=input_shape, weights=[embedding_matrix],
                                       mask_zero=True, trainable=False)(sentence_input)
    word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)
    print('word embeddings shape is :', word_embeddings.shape)

    # Take token markers that identify entity positions, convert to position embeddings
    entity_markers = layers.Input(shape=(input_shape,))
    pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
                                      input_length=input_shape,
                                      mask_zero=True, embeddings_regularizer=regularizers.l2(), trainable=True)(entity_markers)

    # Merge word and position embeddings and apply the specified amount of RNN layers
    x = layers.concatenate([word_embeddings, pos_embeddings])
    for i in range(p["rnn1_layers"] - 1):
        lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
        if p['bidirectional']:
            lstm_layer = layers.Bidirectional(lstm_layer)
        x = lstm_layer(x)

    lstm_layer = layers.LSTM(p['units1'], return_sequences=False)
    if p['bidirectional']:
        lstm_layer = layers.Bidirectional(lstm_layer)
    sentence_vector = lstm_layer(x)

    return models.Model(inputs=[sentence_input, entity_markers], outputs=sentence_vector)


# 计算欧式距离
def euclidean_distance(vects):
    v1, v2 = vects
    return K.sqrt(K.sum(K.square(v1 - v2), axis=1, keepdims=True))


def cosine_distance(vects):
    v1, v2 = vects
    f_x1x2 = tf.reduce_sum(tf.multiply(v1, v2), 1)
    norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(v1), 1))
    norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(v2), 1))
    Ew = f_x1x2 / (norm_fx1 * norm_fx2)
    return Ew


def eucl_dist_output_shape(shapes):
    # 在这里我们需要求修改output_shape, 为(batch, 1)
    shape1, shape2 = shapes
    return (shape1[0], 1)


# 创建contrastive_loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# 创建训练时计算acc的方法
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def model_Siamese(p, embedding_matrix, max_sent_len, n_out):
    print("Parameters:", p)

    sentence_input1 = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input1')
    entity_markers1 = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers1')
    sentence_input2 = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input2')
    entity_markers2 = layers.Input(shape=(max_sent_len,), dtype='int8', name='entity_markers2')

    base_network = create_base_network(max_sent_len, embedding_matrix, p)


    # 获取经过模型后的输出
    processed_a = base_network([sentence_input1, entity_markers1])
    processed_b = base_network([sentence_input2, entity_markers2])


    # # Apply softmax
    sentence_vector = layers.Dropout(p['dropout1'])(processed_a)
    main_output = layers.Dense(n_out, activation="softmax", name='main_output')(sentence_vector)

    model = models.Model(inputs=[sentence_input1, entity_markers1, sentence_input2, entity_markers2], outputs=[main_output])
    print('model layers number is: ', len(model.layers))
    for layer in model.layers[:2]:
        layer.trainable = False
        #print(layer.name)
    model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def to_indices(graphs, word2idx):
    max_sent_len = model_params['max_sent_len']
    num_edges = sum(1 for g in graphs for e in g['edgeSet'])
    sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
    entity_matrix = np.zeros((num_edges, max_sent_len), dtype="int8")
    y_matrix = np.zeros(num_edges, dtype="int16")
    index = 0
    for g in tqdm.tqdm(graphs, ascii=True):
        token_sent_ids = embeddings.get_idx_sequence(g["tokens"], word2idx)
        if len(token_sent_ids) > max_sent_len:
            token_sent_ids = token_sent_ids[:max_sent_len]
        for edge in g["edgeSet"]:
            entity_markers = [m for _, m in graph_utils.get_entity_indexed_vector(g["tokens"], edge, mode=POSITION_EMBEDDING_MODE)]
            sentences_matrix[index, :len(token_sent_ids)] = token_sent_ids
            entity_matrix[index, :len(token_sent_ids)] = entity_markers[:len(token_sent_ids)]
            _, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
            property_kbid = property2idx.get(property_kbid)
            y_matrix[index] = property_kbid
            index += 1
    return [sentences_matrix, entity_matrix, y_matrix]


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
