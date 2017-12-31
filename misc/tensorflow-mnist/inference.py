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


def do_padding(data, padding, weight_shape):
    if padding == 'VALID':
        return data
    assert(padding == 'SAME')
    p_h = int(weight_shape[0] / 2)
    p_w = int(weight_shape[1] / 2)

    assert(len(data.shape) == 4)
    padded_data_shape = list(data.shape)
    padded_data_shape[1] += p_h * 2
    padded_data_shape[2] += p_w * 2
    padded_data = np.empty(padded_data_shape)

    for i in xrange(len(data)):
        img = np.lib.pad(data[i], ((p_h, p_h), (p_w, p_w), (0, 0)), 'constant')
        padded_data[i] = img

    return padded_data


def fprop_conv2d(weights, biases, data, padding='SAME', strides=(1, 1, 1, 1)):
    assert(len(data.shape) == 4)
    assert(len(weights.shape) == 4)
    data = do_padding(data, padding, weights.shape)

    batch_size = data.shape[0]
    in_height = data.shape[1]
    in_width = data.shape[2]
    in_channel = data.shape[3]
    kernel_height = weights.shape[0]
    kernel_width = weights.shape[1]
    out_channel = weights.shape[3]
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    out_data = np.empty((batch_size, out_height, out_width, out_channel))
    for i in xrange(batch_size):
        for c in xrange(out_channel):
            for h in xrange(0, out_height, strides[1]):
                for w in xrange(0, out_width, strides[2]):
                    conv = 0.0
                    for ic in range(in_channel):
                        conv_kernel = weights[:, :, ic, c]
                        data_block = data[i, h:h+kernel_height, w:w+kernel_width, ic]
                        assert conv_kernel.shape == data_block.shape
                        conv += np.inner(data_block.reshape((-1)), conv_kernel.reshape((-1)))
                    out_data[i][h][w][c] = conv

    return out_data


def fprop_relu(data):
    out = data
    return out


def fprop_maxpool(data):
    out = data
    return out


def fprop_softmax(data):
    out = data
    return out


def fprop_fc(weights, biases, data):
    out = data
    return out


def predict(model, data):
    (conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases) = model
    out = fprop_conv2d(conv1_weights, conv1_biases, data)
    out = fprop_relu(out)
    out = fprop_maxpool(out)
    out = fprop_conv2d(conv2_weights, conv2_biases, out)
    out = fprop_relu(out)
    out = fprop_maxpool(out)
    out = fprop_fc(fc1_weights, fc1_biases, out)
    out = fprop_relu(out)
    out = fprop_fc(fc2_weights, fc2_biases, out)
    out = fprop_softmax(out)
    return out


def main():
    test_data_filename = WORK_DIRECTORY + '/t10k-images-idx3-ubyte.gz'
    test_labels_filename = WORK_DIRECTORY + '/t10k-labels-idx1-ubyte.gz'
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    model = load_model('model')

    def eval_in_batches(data):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = predict(model, data[begin:end, ...])
            else:
                batch_predictions = predict(model, data[-EVAL_BATCH_SIZE:, ...])
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    test_error = error_rate(eval_in_batches(test_data), test_labels)
    print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
    main()
