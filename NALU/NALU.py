import tensorflow as tf
import numpy as np
from common.layer import *


# Neural Arithmetic Logic Unit, for learning simple arithmetic operations
# All weights of this neural net are forced to be 1, 0, or -1

# A Neural Accumulator learns addition and subtraction
def NAC_layer(input, output_dim, w_hat_stdev=0.05, m_hat_stdev=0.05):
    shape = tf.concat(tf.shape(input), output_dim)
    W_hat = tf.Variable(tf.random_normal(shape, stddev=w_hat_stdev))
    M_hat = tf.Variable(tf.random_normal(shape, stddev=m_hat_stdev))
    # tanh forces the weights to be 1 or -1, sigmoid forces the weights to be 1, 0, or -1
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    return tf.matmul(input, W)

# A more complex Neural Accumulator can learn multiplication, division and exponents
def complex_NAC_layer(input, output_dim, w_hat_stdev=0.05, m_hat_stdev=0.05, epsilon=10e-8):
    shape = tf.concat(tf.shape(input), output_dim)
    W_hat = tf.Variable(tf.random_normal(shape, stddev=w_hat_stdev))
    M_hat = tf.Variable(tf.random_normal(shape, stddev=m_hat_stdev))
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    # e^(w1log(x1) + w2log(x2)) = x1^w1 * x2^w2, a generalized form for mult, div, exp and log operations
    return tf.exp(tf.matmul(tf.log(tf.abs(input) + epsilon), W))

