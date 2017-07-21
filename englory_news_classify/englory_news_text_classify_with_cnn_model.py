# -*- coding: utf-8 -*-

"""
@Time    : 7/11/17 3:35 PM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : englory_news_text_classify_with_cnn_model.py
@Software: PyCharm
@Description:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf
from sklearn import metrics

FLAGS = None

MAX_DOCUMENT_LENGTH = 300
EMBEDDING_SIZE = 30

N_FILTERS_1st = 16
N_FILTERS_2nd = 32
N_FILTERS_3rd = 64

WINDOW_SIZE = 3
FILTER_SHAPE1 = [WINDOW_SIZE, WINDOW_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, WINDOW_SIZE]
POOLING_WINDOW = 2
POOLING_STRIDE = 2
n_words = 0
MAX_LABEL = 18
WORDS_FEATURE = 'words'  # Name of the input words feature.
BATCH_SIZE = 100

FULL_CONN_NODES_COUNT = 512


def cnn_model(features, labels, mode):
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = tf.layers.conv2d(
            word_vectors,
            filters=N_FILTERS_1st,
            kernel_size=FILTER_SHAPE1,
            padding='SAME',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        # pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = tf.layers.conv2d(
            pool1,
            filters=N_FILTERS_2nd,
            kernel_size=FILTER_SHAPE2,
            padding='SAME')
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    with tf.variable_scope('CNN_Layer3'):
        # Second level of convolution filtering.
        conv3 = tf.layers.conv2d(
            pool2,
            filters=N_FILTERS_3rd,
            kernel_size=FILTER_SHAPE2,
            padding='SAME')
        pool3 = tf.layers.max_pooling2d(
            conv3,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    shape_list = pool3.get_shape().as_list()
    pool3_flat = tf.reshape(pool3, [-1, shape_list[1] * shape_list[2] * shape_list[3]])

    full_conn1 = tf.layers.dense(pool3_flat, FULL_CONN_NODES_COUNT, activation=tf.nn.relu)
    full_conn2 = tf.layers.dense(full_conn1, FULL_CONN_NODES_COUNT, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=full_conn2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Apply regular WX + B and classification.
    logits = tf.layers.dense(dropout, MAX_LABEL, activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    global n_words
    # Prepare training and testing data

    train_file = '/home/wong/Documents/dataset/englory_news/news.data.train.example'
    test_file = '/home/wong/Documents/dataset/englory_news/news.data.test.example'
    train_set = pandas.read_csv(train_file, header=None)
    test_set = pandas.read_csv(test_file, header=None)

    x_train_top = pandas.DataFrame(train_set)
    x_train = x_train_top[1]
    y_train = pandas.Series(x_train_top[0])

    x_test_top = pandas.DataFrame(test_set)
    x_test = x_test_top[1]
    y_test = pandas.Series(x_test_top[0])

    # Process vocabulary
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # Build model
    classifier = tf.estimator.Estimator(model_fn=cnn_model)

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_train},
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=3000)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)

    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_with_fake_data',
        default=False,
        help='Test the example code with fake data.',
        action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
