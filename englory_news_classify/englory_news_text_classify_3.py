# -*- coding: utf-8 -*-

"""
@Time    : 7/21/17 4:10 PM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : englory_news_text_classify_2.py
@Software: PyCharm
@Description:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

np.set_printoptions(threshold=np.inf)

TEXT_LEN = 300
EMBEDDING_LEN = 20

POOL1_DEEP = 32
POOL2_DEEP = 64

CATEGORY_COUNT = 18

n_words = 0


# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_LEN)

    input_layer = tf.expand_dims(word_vectors, 3)
    """Model function for CNN."""
    # Input Layer
    # input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=POOL1_DEEP,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=POOL2_DEEP,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    shape_list = pool2.get_shape().as_list()
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, shape_list[1] * shape_list[2] * shape_list[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=CATEGORY_COUNT)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CATEGORY_COUNT)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    global n_words
    # Prepare training and testing data

    train_file = '/home/wong/Documents/dataset/englory_news/news.data.train.example'
    test_file = '/home/wong/Documents/dataset/englory_news/news.data.test.example'
    train_set = pandas.read_csv(train_file, header=None)
    test_set = pandas.read_csv(test_file, header=None)

    x_train_top = pandas.DataFrame(train_set)
    x_train = x_train_top[1]
    # y_train = pandas.Series(x_train_top[0])
    y_train = np.asarray(x_train_top[0], dtype=np.int32)

    x_test_top = pandas.DataFrame(test_set)
    x_test = x_test_top[1]
    # y_test = pandas.Series(x_test_top[0])
    y_test = np.asarray(x_test_top[0], dtype=np.int32)

    # Process vocabulary
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        TEXT_LEN)

    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # Load training and eval data
    # mnist = learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/text_classify_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
        x=x_train,
        y=y_train,
        batch_size=100,
        steps=2000000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=x_test, y=y_test
        , metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
