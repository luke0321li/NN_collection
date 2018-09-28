import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from common.layer import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class VAE:
    def __init__(self, batch_size=20, learning_rate=10e-3, iterations=1000, compress_dim=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.compress_dim = compress_dim

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.x_flat = tf.reshape(self.x, [-1, 28 * 28])

        # Fully connected layer
        self.w_1, self.b_1 = make_w_and_b([28 * 28, 28 * 28])
        self.fc_1 = tf.nn.relu(tf.matmul(self.x_flat, self.w_1) + self.b_1)

        # Another fully connected layer creating mean and std
        self.w_mean, self.b_mean = make_w_and_b([28 * 28, self.compress_dim])
        # self.mean = tf.nn.relu(tf.matmul(self.fc_1, self.w_mean) + self.b_mean)
        self.mean = tf.matmul(self.fc_1, self.w_mean) + self.b_mean

        self.w_stdev, self.b_stdev = make_w_and_b([28 * 28, self.compress_dim])
        # Softplus is used to force standard deviation to be positive
        self.stdev = tf.nn.softplus(tf.matmul(self.fc_1, self.w_stdev) + self.b_stdev)

        # Sampling from a normal distribution
        self.normal = tf.random_normal([self.batch_size, self.compress_dim], 0.0, 1.0, dtype=tf.float32)
        self.sample = self.mean + self.normal * self.stdev

        # Decoder
        # Another fully connected layer
        self.w_2, self.b_2 = make_w_and_b([self.compress_dim, 28 * 28])
        self.fc_2 = tf.nn.relu(tf.matmul(self.sample, self.w_2) + self.b_2)

        # Another fully connected layer reconstructing output image
        self.w_out, self.b_out = make_w_and_b([28 * 28, 28 * 28])
        self.output_flat = tf.nn.sigmoid(tf.matmul(self.fc_2, self.w_out) + self.b_out)
        self.output_image = tf.reshape(self.output_flat, [self.batch_size, 28, 28, 1])

        # Cost functions
        self.divergence = 0.5 * tf.reduce_sum(
            tf.square(self.mean) + tf.square(self.stdev) - tf.log(1e-8 + self.stdev) - 1.0, 1)

        # Note: l2_loss is the only working metric so far
        self.difference = tf.nn.l2_loss(self.x_flat - self.output_flat)

        self.cost = tf.reduce_mean(self.divergence + self.difference)
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
                # cost = self.cost.eval(feed_dict={self.x: train_x})
                # print(np.mean(cost))

                if i % 100 == 0:
                    print("%d iterations complete" % i)
                    # test_sample = self.output_flat.eval(feed_dict={self.x: train_x})[0].reshape([28, 28])
                    # plt.gray()
                    # plt.imshow((test_sample * 255).astype(np.uint8), interpolation='nearest')
                    # plt.show()

            # Randomly generate some samples
            for i in range(10):
                test_sample = self.output_flat.eval(
                    feed_dict={self.sample: np.random.normal(0, 1, (self.batch_size, self.compress_dim))})[0].reshape([28, 28])
                plt.gray()
                plt.imshow((test_sample * 255).astype(np.uint8), interpolation='nearest')
                plt.show()


if __name__ == '__main__':
    VAE = VAE(100, 0.001, 10000, 50)
    VAE.train_mnist()
