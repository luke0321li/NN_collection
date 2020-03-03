import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from common.layer import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class AE:
    def __init__(self, input_dim=10, intermediate_dim=5, compress_dim=5, batch_size=20, learning_rate=10e-3, iterations=1000):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.compress_dim = compress_dim

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.mask = tf.placeholder(tf.float32, [None, self.input_dim])

        # Encoder
        # 2 fully connected layers
        self.w_1, self.b_1 = make_w_and_b([self.input_dim, self.intermediate_dim])
        self.fc_1 = tf.nn.relu(tf.matmul(self.x, self.w_1) + self.b_1)

        self.w_2, self.b_2 = make_w_and_b([self.intermediate_dim, self.compress_dim])
        self.compressed = tf.nn.relu(tf.matmul(self.fc_1, self.w_2) + self.b_2)

        # Decoder
        # Another 2 fully connected layers
        self.w_3, self.b_3 = make_w_and_b([self.compress_dim, self.intermediate_dim])
        self.fc_2 = tf.nn.relu(tf.matmul(self.compressed, self.w_3) + self.b_3)

        self.w_4, self.b_4 = make_w_and_b([self.intermediate_dim, self.input_dim])
        self.output = tf.nn.relu(tf.matmul(self.fc_2, self.w_4) + self.b_4)

        # Loss function
        self.cost = tf.nn.l2_loss(self.x - self.output * self.mask)
        self.objective = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def train_mnist(self):
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.iterations):
                batch = mnist.train.next_batch(self.batch_size)
                train_x = batch[0].reshape([self.batch_size, 28 * 28 * 1])
                # train_y = batch[1]
                train_m = np.random.binomial(1, 0.6, train_x.shape)

                sess.run(self.objective, feed_dict={self.x: train_x, self.mask: train_m})
                cost = self.cost.eval(feed_dict={self.x: train_x, self.mask: train_m})
                print(np.mean(cost))

                if i % 1000 == 0:
                    print("%d iterations complete" % i)
                    test_sample = self.output.eval(feed_dict={self.x: train_x, self.mask: train_m})[-1].reshape([28, 28])
                    plt.gray()
                    plt.imshow((test_sample * 255).astype(np.uint8), interpolation='nearest')
                    plt.savefig("%d.png" % i)
                    plt.clf()
                    plt.gray()
                    plt.imshow((train_x[-1] * train_m[-1] * 255).reshape(28, 28).astype(np.uint8), interpolation='nearest')
                    plt.savefig("%d_true.png" % i)
                    plt.gray()


    def train(self, data, masks):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        data_size = len(data)
        costs = np.zeros(self.iterations)
        with tf.Session() as sess:
            sess.run(init)
            current_index = 0
            for i in range(self.iterations):
                batch = data.take(np.arange(current_index, current_index + self.batch_size), axis=0, mode='wrap')
                mask = masks.take(np.arange(current_index, current_index + self.batch_size), axis=0, mode='wrap')
                current_index += self.batch_size
                sess.run(self.objective, feed_dict={self.x: batch, self.mask: mask})
                cost = self.cost.eval(feed_dict={self.x: batch, self.mask: mask})
                costs[i] = cost
                if i % 1000 == 0:
                    print(cost)
            path = saver.save(sess, "model_1")
            print(path)
        return costs


if __name__ == '__main__':
    AE = AE(input_dim = 28 * 28, intermediate_dim=100, compress_dim=10, batch_size=100, learning_rate=0.01, iterations=10000)
    AE.train_mnist()
