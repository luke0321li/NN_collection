import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# def default_layer(input, in_size, out_size, activation_function=None):
#     w = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.1))
#     b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#     z = tf.matmul(input, w) + b
#     if activation_function is None:
#         output = z
#     else:
#         output = activation_function(z)
#     return output


class CNN:
    def __init__(self, batch_size, learning_rate, iterations):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # Convolutional layer 1
        self.w1, self.b1 = make_w_and_b([5, 5, 1, 6])
        # In other words: use convolutional filters of size 5x5, from 1 channel to 6 channels (in total, 6 filters)
        self.conv_1 = conv2d_biased_layer(self.x, self.w1, self.b1, 'SAME')

        # Pooling 1
        self.pool_1 = pool_layer(self.conv_1, 2, 'VALID')

        # Convolutional layer 2
        self.w2, self.b2 = make_w_and_b([5, 5, 6, 16])
        self.conv_2 = conv2d_biased_layer(self.pool_1, self.w2, self.b2, 'VALID')

        # Pooling 2
        self.pool_2 = pool_layer(self.conv_2, 2, 'VALID')

        # Fully connected layer 1
        self.w3, self.b3 = make_w_and_b([5 * 5 * 16, 120])
        self.pool_2_to_fc_1 = tf.reshape(self.pool_2, [-1, 5 * 5 * 16])
        self.fc_1 = tf.nn.relu(tf.matmul(self.pool_2_to_fc_1, self.w3) + self.b3)

        # Fully connected layer 2
        self.w4, self.b4 = make_w_and_b([120, 84])
        self.fc_2 = tf.nn.relu(tf.matmul(self.fc_1, self.w4) + self.b4)

        # Fully connected layer 3
        self.w5, self.b5 = make_w_and_b([84, 10])
        self.output = tf.nn.softmax(tf.matmul(self.fc_2, self.w5) + self.b5)

        # Cost function
        self.cost = tf.nn.l2_loss(self.output - self.y)

        # Objective
        self.objective = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.output, axis=1), tf.argmax(self.y, axis=1)), tf.float32))

    def train_mnist(self):
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.iterations):
                batch = mnist.train.next_batch(self.batch_size)
                train_x = batch[0].reshape([self.batch_size, 28, 28, 1])
                train_y = batch[1]

                sess.run(self.objective, feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print("%d iterations complete" % i)

            final_accuracy = self.accuracy.eval(
                feed_dict={self.x: mnist.test.images.reshape([-1, 28, 28, 1]), self.y: mnist.test.labels})

            print("Accuracy: ")
            print(final_accuracy)


CNN = CNN(50, 0.01, 1000)
CNN.train_mnist()
