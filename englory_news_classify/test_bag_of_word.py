# -*- coding: utf-8 -*-

"""
@Time    : 7/25/17 10:37 AM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : test_bag_of_word.py
@Software: PyCharm
@Description:
"""

import numpy as np
import pandas
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

MAX_DOCUMENT_LENGTH = 300
EMBEDDING_SIZE = 20


def rnn_model(features, target):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and
    # then maps word indexes of the sequence into [batch_size,
    # sequence_length, EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
    # Split into list of embedding per word, while removing doc length
    # dim. word_list results to be a list of tensors [batch_size,
    # EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each
    # unit.
    # _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)
    _, encoding = tf.contrib.rnn(cell, word_list, dtype=tf.float32)
    # Given encoding of RNN, take encoding of last step (e.g hidden
    # size of the neural network of last step) and pass it as features
    # to fully connected layer to output probabilities per class.

    target = tf.one_hot(target, 15, 1, 0)
    logits = tf.contrib.layers.fully_connected(
        encoding, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer='Adam', learning_rate=0.01, clip_gradients=1.0)
    return (
        {'class': tf.argmax(logits, 1),
         'prob': tf.nn.softmax(logits)},
        loss, train_op)


def bag_of_words_model(features, target):
    """A bag-of-words model. Note it disregards the word order in the text."""
    target = tf.one_hot(target, 15, 1, 0)
    features = tf.contrib.layers.bow_encoder(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    logits = tf.contrib.layers.fully_connected(features, 15,
                                               activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer='Adam', learning_rate=0.001)
    return {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op


# Prepare training and testing data
dbpedia = learn.datasets.load_dataset('dbpedia')
x_train = pandas.DataFrame(dbpedia.train.data)[1]
y_train = pandas.Series(dbpedia.train.target)
x_test = pandas.DataFrame(dbpedia.test.data)[1]
y_test = pandas.Series(dbpedia.test.target)
# Process vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(
    MAX_DOCUMENT_LENGTH)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

classifier = learn.Estimator(model_fn=rnn_model)
# Train and predict
classifier.fit(x_train, y_train, steps=100)
y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
# print(y_predicted)
# print(y_test)
score = metrics.accuracy_score(y_test, y_predicted)

print('Accuracy: {0: f}'.format(score))
