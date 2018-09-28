import tensorflow as tf
import numpy as np
from common.layer import *


# Neural Arithmetic Logic Unit, for learning simple arithmetic operations
# All weights of this neural net are forced to be 1, 0, or -1

# A Neural Accumulator learns addition and subtraction
def NAC_layer(input, in_dim, out_dim, W_in=None, M_in=None):
    shape = [in_dim, out_dim]
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not W_in else W_in
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not M_in else M_in
    # tanh forces the weights to be 1 or -1, sigmoid forces the weights to be 1, 0, or -1
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    return tf.matmul(input, W)


# A complex Neural Accumulator can learn multiplication, division and exponents
def complex_NAC_layer(input, in_dim, out_dim, W_in=None, M_in=None):
    shape = [in_dim, out_dim]
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not W_in else W_in
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05)) if not M_in else M_in
    W = tf.multiply(tf.nn.tanh(W_hat), tf.nn.sigmoid(M_hat))
    # e^(w1log(x1) + w2log(x2)) = x1^w1 * x2^w2, a generalized form for mult, div, exp and log operations
    return tf.exp(tf.matmul(tf.log(tf.abs(input) + 10e-8), W))


# The full NALU learns a weighted sum of a neural accumulator and a complex neural accumulator with shared weights
def NALU_layer(input, in_dim, out_dim):
    shape = [in_dim, out_dim]
    W_hat = tf.Variable(tf.random_normal(shape, stddev=0.05))
    M_hat = tf.Variable(tf.random_normal(shape, stddev=0.05))
    G = tf.Variable(tf.random_normal(shape, stddev=0.05))
    # g controls the on/off of the two cells
    g = tf.nn.sigmoid(tf.matmul(input, G))
    a = NAC_layer(input, in_dim, out_dim, W_hat, M_hat)
    m = complex_NAC_layer(input, in_dim, out_dim, W_hat, M_hat)
    return tf.multiply(g, a) + tf.multiply(1 - g, m)


# The main wrapper
class NALU:
    def __init__(self, in_dim, out_dim, batch_size=10, learning_rate=10e-3, iterations=10000):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.x = tf.placeholder(tf.float32, [None, in_dim])
        self.y = tf.placeholder(tf.float32, [None, out_dim])

        self.output = NALU_layer(self.x, in_dim, out_dim)

        # Cost function
        self.cost = tf.nn.l2_loss(self.output - self.y)

        # Objective
        self.objective = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def train(self, x, y, plot=False):
        data = np.append(y, x, axis=1)
        np.random.shuffle(data)
        index = 0
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(self.iterations):
                if index + self.batch_size >= data.shape[0]:
                    np.random.shuffle(data)
                    index = 0
                train_x = data[index:index + self.batch_size, 1:]
                train_y = np.reshape(data[index:index + self.batch_size, 0], (self.batch_size, y.shape[1]))
                index += self.batch_size
                sess.run(self.objective, feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print("%d iterations complete" % i)

                if i % 1000 == 0:
                    print("Current cost: ")
                    print(self.cost.eval(
                        feed_dict={self.x: data[:, 1:], self.y: np.reshape(data[:, 0], (data.shape[0], 1))}))
                    
            final_cost = self.cost.eval(
                feed_dict={self.x: data[:, 1:], self.y: np.reshape(data[:, 0], (data.shape[0], 1))})

            print("Cost: ")
            print(final_cost)


if __name__ == '__main__':
    x = np.random.randn(3000, 5)
    y = x[:, 0] * x[:, 2] + 3 * x[:, 4] / x[:, 1] - x[:, 3]
    y = x[:, 0] + x[:, 1]
    y = np.reshape(y, (3000, 1))
    NALU = NALU(5, 1)
    NALU.train(x, y)
