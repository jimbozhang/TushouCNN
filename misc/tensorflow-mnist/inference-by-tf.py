# Copyright 2017-2018. All Rights Reserved.
# Author: Junbo Zhang <dr.jimbozhang@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This script inferences with the saved model for MNIST."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
EVAL_BATCH_SIZE = 64


def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def error_rate(predictions, labels):
    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 1) == labels) /
            predictions.shape[0])


def load_model(modeldir):
    return (np.loadtxt(modeldir + '/0_conv1_weights', 'float32').reshape([5, 5, NUM_CHANNELS, 32]),
            np.loadtxt(modeldir + '/1_conv1_biases', 'float32').reshape([32]),
            np.loadtxt(modeldir + '/2_conv2_weights', 'float32').reshape([5, 5, 32, 64]),
            np.loadtxt(modeldir + '/3_conv2_biases', 'float32').reshape([64]),
            np.loadtxt(modeldir + '/4_fc1_weights', 'float32').reshape(
                [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
            np.loadtxt(modeldir + '/5_fc1_biases', 'float32').reshape([512]),
            np.loadtxt(modeldir + '/6_fc2_weights', 'float32').reshape([512, NUM_LABELS]),
            np.loadtxt(modeldir + '/7_fc2_biases', 'float32').reshape([NUM_LABELS]))


def main():
    (conv1_weights, conv1_biases, conv2_weights, conv2_biases,
     fc1_weights, fc1_biases, fc2_weights, fc2_biases) = load_model('model')

    def model(data):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    eval_prediction = tf.nn.softmax(model(eval_data))

    test_data_filename = WORK_DIRECTORY + '/t10k-images-idx3-ubyte.gz'
    test_labels_filename = WORK_DIRECTORY + '/t10k-labels-idx1-ubyte.gz'
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
    main()
