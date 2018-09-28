import tensorflow as tf
import numpy as np
from common.layer import *


# Neural Arithmetic Logic Unit, for learning simple arithmetic operations
# All weights of this neural net are forced to be 1, 0, or -1

# A Neural Accumulator learns addition and subtraction
def NAC_layer(input, output_dim, W_in=None, M_in=None):
    shape = tf.concat(tf.shape(input)[1:], output_dim)
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not W_in else W_in
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not M_in else M_in
    # tanh forces the weights to be 1 or -1, sigmoid forces the weights to be 1, 0, or -1
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    return tf.matmul(input, W)


# A complex Neural Accumulator can learn multiplication, division and exponents
def complex_NAC_layer(input, output_dim, W_in=None, M_in=None):
    shape = tf.concat(tf.shape(input)[1:], output_dim)
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not W_in else W_in
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not M_in else M_in
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    # e^(w1log(x1) + w2log(x2)) = x1^w1 * x2^w2, a generalized form for mult, div, exp and log operations
    return tf.exp(tf.matmul(tf.log(tf.abs(input) + 10e-8), W))


# The full NALU learns a weighted sum of a neural accumulator and a complex neural accumulator with shared weights
def NALU(input, output_dim):
    shape = tf.concat(tf.shape(input)[1:], output_dim)
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05))
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05))
    G = tf.Variable(tf.random_normal(shape, stddev=0.05))
    # g controls the on/off of the two cells
    g = tf.nn.sigmoid(tf.matmul(input, G))
    a = NAC_layer(input, output_dim, W_hat, M_hat)
    m = complex_NAC_layer(input, output_dim, W_hat, M_hat)
    return tf.multiply(g, a) + tf.multiply(1 - g, m)

