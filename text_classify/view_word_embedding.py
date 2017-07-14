# -*- coding: utf-8 -*-

"""
@Time    : 7/12/17 1:45 PM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : test_view_word_embedding.py
@Software: PyCharm
@Description: 词向量的生成。使用numpy对词进行统计并编码。
"""

import numpy as np
import pandas
import tensorflow as tf

EMBEDDING_SIZE = 20
MAX_DOCUMENT_LENGTH = 50
n_words = 0

dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')

x_train_top = pandas.DataFrame(dbpedia.train.data)
x_train = x_train_top[1]
y_train = pandas.Series(dbpedia.train.target)

x_test_top = pandas.DataFrame(dbpedia.test.data)
x_test = x_test_top[1]
y_test = pandas.Series(dbpedia.test.target)

# Process vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
    MAX_DOCUMENT_LENGTH)

x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.transform(x_test)))

n_words = len(vocab_processor.vocabulary_)

print('Total words: %d' % n_words)
word_vectors = tf.contrib.layers.embed_sequence(
    x_train, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
word_vectors = tf.expand_dims(word_vectors, 3)
print('Finish!')
