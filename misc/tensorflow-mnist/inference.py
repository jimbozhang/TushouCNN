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
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
EVAL_BATCH_SIZE = 64

FLAGS = None

def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels

def error_rate(predictions, labels):
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == labels) /
            predictions.shape[0])

def load_model(modeldir):
    variables = []
    variables.append(numpy.loadtxt(modeldir + '/0_conv1_weights', 'float32').reshape([5, 5, NUM_CHANNELS, 32]))
    variables.append(numpy.loadtxt(modeldir + '/1_conv1_biases', 'float32').reshape([32]))
    variables.append(numpy.loadtxt(modeldir + '/2_conv2_weights', 'float32').reshape([5, 5, 32, 64]))
    variables.append(numpy.loadtxt(modeldir + '/3_conv2_biases', 'float32').reshape([64]))
    variables.append(numpy.loadtxt(modeldir + '/4_fc1_weights', 'float32').reshape([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]))
    variables.append(numpy.loadtxt(modeldir + '/5_fc1_biases', 'float32').reshape([512]))
    variables.append(numpy.loadtxt(modeldir + '/6_fc2_weights', 'float32').reshape([512, NUM_LABELS]))
    variables.append(numpy.loadtxt(modeldir + '/7_fc2_biases', 'float32').reshape([NUM_LABELS]))
    return variables

if __name__ == '__main__':
    #[conv1_weights, conv1_biases, conv2_weights, conv2_biases,
    # fc1_weights , fc1_biases, fc2_weights, fc2_biases] = load_model('model')

    test_data_filename = WORK_DIRECTORY + '/t10k-images-idx3-ubyte.gz'
    test_labels_filename = WORK_DIRECTORY + '/t10k-labels-idx1-ubyte.gz'
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    def eval_in_batches(data):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
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

    test_error = error_rate(eval_in_batches(test_data), test_labels)
    print('Test error: %.1f%%' % test_error)
