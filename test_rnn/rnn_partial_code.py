# -*- coding: utf-8 -*-

"""
@Time    : 7/27/17 2:47 PM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : rnn_partial_code.py
@Software: PyCharm
@Description:
"""
import tensorflow as tf
from tensorflow.contrib import rnn
lstm=rnn.BasicLSTMCell(500)
stacked_lstm=rnn.MultiRNNCell([lstm]*3)

state=stacked_lstm.zero_state(100,tf.float32)

