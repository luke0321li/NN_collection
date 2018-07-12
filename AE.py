import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from layer import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class AE:
    def __init__(self, batch_size, learning_rate, iterations, compress_dim):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.compress_dim = compress_dim

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])

        # Encoder
        # 2 fully connected layers
        self.x_flat = tf.reshape(self.x, [self.batch_size, 28 * 28])
        self.w_1, self.b_1 = make_w_and_b([28 * 28, 10 * 10])
        self.fc_1 = tf.nn.relu(tf.matmul(self.x_flat, self.w_1) + self.b_1)

        self.w_2, self.b_2 = make_w_and_b([10 * 10, self.compress_dim])
        self.compressed = tf.nn.relu(tf.matmul(self.fc_1, self.w_2) + self.b_2)

        # Decoder
        # Another 2 fully connected layers
        self.w_3, self.b_3 = make_w_and_b([self.compress_dim, 10 * 10])
        self.fc_2 = tf.nn.relu(tf.matmul(self.compressed, self.w_3) + self.b_3)

        self.w_4, self.b_4 = make_w_and_b([10 * 10, 28 * 28])
        self.output = tf.nn.sigmoid(tf.matmul(self.fc_2, self.w_4) + self.b_4)
        self.output_image = tf.reshape(self.output, [self.batch_size, 28, 28, 1])

        # Loss function
        self.cost = tf.nn.l2_loss(self.x_flat - self.output)
        self.objective = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def train_mnist(self):
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.iterations):
                batch = mnist.train.next_batch(self.batch_size)
                train_x = batch[0].reshape([self.batch_size, 28, 28, 1])
                # train_y = batch[1]

                sess.run(self.objective, feed_dict={self.x: train_x})
                cost = self.cost.eval(feed_dict={self.x: train_x})
                print(np.mean(cost))

                if i % 1000 == 0:
                    print("%d iterations complete" % i)
                    test_sample = self.output.eval(feed_dict={self.x: train_x})[-1].reshape([28, 28])
                    plt.gray()
                    plt.imshow((test_sample * 255).astype(np.uint8), interpolation='nearest')
                    plt.show()


AE = AE(100, 0.01, 10000, 10)
AE.train_mnist()
