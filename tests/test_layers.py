# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
import os
import unittest

import numpy as np
import keras
import onnx
import keras2onnx
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from distutils.version import StrictVersion
import tf2onnx


working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')


class TestKerasTF2ONNX(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @staticmethod
    def asarray(*a):
        return np.array([a], dtype='f')

    @staticmethod
    def get_temp_file(name):
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        return os.path.join(tmp_path, name)

    def run_onnx_runtime(self, case_name, onnx_model, data, expected, rtol=1.e-4, atol=1.e-8):
        temp_model_file = TestKerasTF2ONNX.get_temp_file('temp_' + case_name + '.onnx')
        onnx.save_model(onnx_model, temp_model_file)
        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(temp_model_file)
        except ImportError:
            return True

        if not isinstance(expected, list):
            expected = [expected]

        data = data if isinstance(data, list) else [data]
        feed = dict([(x.name, data[n]) for n, x in enumerate(sess.get_inputs())])
        actual = sess.run(None, feed)
        res = all(np.allclose(expected[n_], actual[n_], rtol=rtol, atol=atol) for n_ in range(len(expected)))
        if res and temp_model_file not in self.model_files:  # still keep the failed case files for the diagnosis.
            self.model_files.append(temp_model_file)

        return res

    def test_keras_lambda(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Lambda(lambda x: x ** 2, input_shape=[3, 5]))
        model.add(keras.layers.Flatten(data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')

        onnx_model = keras2onnx.convert_keras(model, 'test')
        data = np.random.rand(3 * 5).astype(np.float32).reshape(1, 3, 5)
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('onnx_lambda', onnx_model, data, expected))

    def test_dense(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(5, input_shape=(4,), activation='sigmoid'))
        model.add(keras.layers.Dense(3, input_shape=(5,), use_bias=True))
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = self.asarray(1, 0, 0, 1)
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('dense', onnx_model, data, expected))

    def test_dense_add(self):
        input1 = keras.layers.Input(shape=(4,))
        x1 = keras.layers.Dense(3, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(5,))
        x2 = keras.layers.Dense(3, activation='sigmoid')(input2)
        input3 = keras.layers.Input(shape=(3,))
        x3 = keras.layers.Dense(3)(input3)
        added = keras.layers.Add()([x1, x2, x3])  # equivalent to added = keras.layers.add([x1, x2])
        model = keras.models.Model(inputs=[input1, input2, input3], outputs=added)
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = [self.asarray(1.2, 2.4, -2, 1), self.asarray(-1, -2, 0, 1, 2), self.asarray(0.5, 1.5, -3.14159)]
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('onnx_dense_add', onnx_model, data, expected))

    def test_dense_softmax(self):
        data = self.asarray(1, 2, 3, 4)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(5, input_shape=(4,), activation='softmax'))
        model.add(keras.layers.Dense(3, input_shape=(5,), use_bias=True))
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('dense_softmax_1', onnx_model, data, expected))

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(5, input_shape=(4,)))
        model.add(keras.layers.Activation('softmax'))
        model.add(keras.layers.Dense(3, input_shape=(5,), use_bias=True))
        model.compile('sgd', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('dense_softmax_2', onnx_model, data, expected))

    def mergelayer_helper(self, keras_layer_type, *data):
        data2 = [self.asarray(*d) for d in data]
        inputs = [keras.layers.Input(shape=d.shape[1:]) for d in data2]
        layer = keras_layer_type()(inputs)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data2)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data2, expected))

    def test_add(self):
        self.mergelayer_helper(keras.layers.Add, [1, 2, 3], [4, 5, 6])
        self.mergelayer_helper(keras.layers.Add, [1, 2, 3], [4, 5, 6], [-3, -1, 1.5])

    def test_sub(self):
        self.mergelayer_helper(keras.layers.Subtract, [1, 2, 3], [4, 5, 6])

    def test_mul(self):
        self.mergelayer_helper(keras.layers.Multiply, [1, 2, 3], [4, 5, 6])

    def test_average(self):
        self.mergelayer_helper(keras.layers.Average, [1, -2, 3], [3, 1, 1])

    def test_max(self):
        self.mergelayer_helper(keras.layers.Maximum, [1, -2, 3], [3, 1, 1])

    def test_concat(self):
        self.mergelayer_helper(lambda: keras.layers.Concatenate(), [1, 2, 3], [4, 5, 6, 7])
        self.mergelayer_helper(lambda: keras.layers.Concatenate(), [1, 2, 3], [4, 5, 6, 7])

    def test_concat_2d(self):
        self.mergelayer_helper(lambda: keras.layers.Concatenate(-1), [[1, 2], [3, 4]], [[4, 5], [6, 7]])
        self.mergelayer_helper(lambda: keras.layers.Concatenate(1), [[1, 2], [3, 4]], [[4, 5], [6, 7]])
        self.mergelayer_helper(lambda: keras.layers.Concatenate(2), [[1, 2], [3, 4]], [[4, 5], [6, 7]])

    def _conv_helper(self, layer_type, input_channels, output_channels, kernel_size, strides, input_size, activation,
                     rtol, atol, bias, channels_first=False, padding='valid'):
        model = keras.Sequential()
        input_size_seq = (input_size,) if isinstance(input_size, int) else input_size
        kwargs = {}
        if channels_first:
            input_shape = (input_channels,) + input_size_seq
            if not isinstance(layer_type, keras.layers.Conv1D):
                kwargs['data_format'] = 'channels_first'
        else:
            input_shape = input_size_seq + (input_channels,)

        model.add(layer_type(output_channels, kernel_size, input_shape=input_shape, strides=strides, padding=padding,
                             dilation_rate=1, activation=activation, use_bias=bias, **kwargs))
        data = np.random.uniform(-0.5, 0.5, size=(1,) + input_shape).astype(np.float32)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected, rtol=rtol, atol=atol))

    def _conv1_helper(self, input_channels, output_channels, kernel_size, strides, input_length, activation=None,
                      rtol=1e-4, atol=1e-6, bias=False, padding='valid'):
        self._conv_helper(keras.layers.Conv1D, input_channels, output_channels, kernel_size, strides, input_length,
                          activation, rtol, atol, bias, padding=padding)

    def test_conv1d(self):
        self._conv1_helper(4, 5, 3, 1, 15)
        self._conv1_helper(4, 5, 3, 2, 15)

    def test_conv1d_padding_same(self):
        self._conv1_helper(4, 5, 3, 1, 15, padding='same')
        # Not sure about 'causal'.

    def test_conv1d_activation(self):
        self._conv1_helper(4, 5, 3, 1, 15, activation='sigmoid')

    def test_conv1d_bias(self):
        self._conv1_helper(4, 5, 3, 1, 15, bias=True)

    def _conv2_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                      rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 2)
        self._conv_helper(keras.layers.Conv2D, input_channels, output_channels, kernel_size, strides, inputs_dims,
                          activation, rtol, atol, bias, channels_first, padding)

    def _conv2trans_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                           rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 2)
        self._conv_helper(keras.layers.Conv2DTranspose, input_channels, output_channels, kernel_size, strides,
                          inputs_dims, activation, rtol, atol, bias, channels_first, padding)

    def test_conv2d(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5))

    def test_conv2d_transpose(self):
        self._conv2trans_helper(3, 5, (2, 2), (1, 1), (5, 5))

    def test_conv2d_padding_same(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), padding='same')

    def test_conv2d_format(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), channels_first=True)

    def test_conv2d_activation(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), activation='relu')

    def test_conv2d_bias(self):
        self._conv2_helper(3, 5, (2, 2), (1, 1), (5, 5), bias=True)

    def test_conv2d_larger(self):
        self._conv2_helper(3, 5, (7, 9), 1, (30, 20))

    def test_conv2d_uneven_stride(self):
        self._conv2_helper(3, 5, (4, 4), (3, 2), (20, 10))

    def _conv3_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                      rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 3)
        self._conv_helper(keras.layers.Conv3D, input_channels, output_channels, kernel_size, strides, inputs_dims,
                          activation, rtol, atol, bias, channels_first, padding)

    def test_conv3d(self):
        self._conv3_helper(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))

    def _conv3trans_helper(self, input_channels, output_channels, kernel_size, strides, inputs_dims, activation=None,
                           rtol=1e-3, atol=1e-5, bias=False, channels_first=False, padding='valid'):
        assert (len(inputs_dims) == 3)
        self._conv_helper(keras.layers.Conv3DTranspose, input_channels, output_channels, kernel_size, strides,
                          inputs_dims, activation, rtol, atol, bias, channels_first, padding)

    @unittest.skip("ONNXRuntime doesn't support 3D ConvTranspose.")
    def test_conv3d_transpose(self):
        self._conv3trans_helper(3, 5, (2, 2, 2), (1, 1, 1), (5, 5, 8))

    def test_flatten(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Flatten(input_shape=(3, 2)))
        model.add(keras.layers.Dense(3))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('flatten', onnx_model, data, expected))

    def test_reshape(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Reshape((2, 3), input_shape=(3, 2)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('reshape', onnx_model, data, expected))

    def test_permute(self):
        model = keras.Sequential()
        model.add(keras.layers.core.Permute((2, 1), input_shape=(3, 2)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([[[1, 2], [3, 4], [5, 6]]]).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('permute', onnx_model, data, expected))

    def test_repeat_vector(self):
        model = keras.Sequential()
        model.add(keras.layers.core.RepeatVector(3, input_shape=(4,)))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = self.asarray(1, 2, 3, 4)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('repeat_vector', onnx_model, data, expected))

    def _pooling_test_helper(self, layer, ishape):
        model = keras.Sequential()
        nlayer = layer(input_shape=ishape) if \
            (layer.__name__.startswith("Global")) else layer(2, input_shape=ishape)

        model.add(nlayer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.random.uniform(-0.5, 0.5, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    @unittest.skip("ONNXRuntime doesn't support 3D average pooling yet.")
    def test_pooling_avg3d(self):
        self._pooling_test_helper(keras.layers.AveragePooling3D, (4, 4, 4, 3))

    def test_pooling_max1d(self):
        self._pooling_test_helper(keras.layers.MaxPool1D, (4, 6))

    def test_pooling_global(self):
        self._pooling_test_helper(keras.layers.GlobalAveragePooling2D, (4, 6, 2))

    def activationlayer_helper(self, layer, data_for_advanced_layer=None):
        if data_for_advanced_layer is None:
            data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
            layer = keras.layers.Activation(layer, input_shape=(data.size,))
        else:
            data = data_for_advanced_layer

        model = keras.Sequential()
        model.add(layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_tanh(self):
        self.activationlayer_helper('tanh')
        self.activationlayer_helper(keras.activations.tanh)

    def test_sigmoid(self):
        self.activationlayer_helper('sigmoid')
        self.activationlayer_helper(keras.activations.sigmoid)

    def test_hard_sigmoid(self):
        self.activationlayer_helper('hard_sigmoid')
        self.activationlayer_helper(keras.activations.hard_sigmoid)

    def test_relu(self):
        self.activationlayer_helper('relu')
        self.activationlayer_helper(keras.activations.relu)

    def test_elu(self):
        self.activationlayer_helper('elu')
        self.activationlayer_helper(keras.activations.elu)

    def test_selu(self):
        self.activationlayer_helper('selu')
        self.activationlayer_helper(keras.activations.selu)

    def test_softsign(self):
        self.activationlayer_helper('softsign')
        self.activationlayer_helper(keras.activations.softsign)

    def test_softplus(self):
        self.activationlayer_helper('softplus')
        self.activationlayer_helper(keras.activations.softplus)

    def test_softmax(self):
        self.activationlayer_helper('softmax')
        self.activationlayer_helper(keras.activations.softmax)

    def test_linear(self):
        self.activationlayer_helper('linear')
        self.activationlayer_helper(keras.activations.linear)

    def test_LeakyRelu(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = keras.layers.advanced_activations.LeakyReLU(alpha=0.1, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_ThresholdedRelu(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = keras.layers.advanced_activations.ThresholdedReLU(theta=1.0, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_ELU(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = keras.layers.advanced_activations.ELU(alpha=1.0, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_PReLU(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)
        layer = keras.layers.advanced_activations.PReLU(alpha_initializer='ones', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)
        layer = keras.layers.advanced_activations.PReLU(alpha_initializer='RandomNormal', input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def test_Softmax(self):
        data = self.asarray(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        layer = keras.layers.advanced_activations.Softmax(axis=-1, input_shape=(data.size,))
        self.activationlayer_helper(layer, data)

    def _misc_conv_helper(self, layer, ishape):
        ishape = (20, 20, 1)
        input = keras.Input(ishape)
        out = layer(input)
        model = keras.models.Model(input, out)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.random.uniform(-0.5, 0.5, size=(1,) + ishape).astype(np.float32)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_crop(self):
        ishape = (20, 20, 1)
        layer = keras.layers.Cropping2D(cropping=((1, 2), (2, 3)), data_format='channels_last')
        self._misc_conv_helper(layer, ishape)

    def test_upsample(self):
        ishape = (20, 20, 1)
        layer = keras.layers.UpSampling2D(size=(2, 3), data_format='channels_last')
        self._misc_conv_helper(layer, ishape)

    def test_padding(self):
        ishape = (20, 20, 1)
        layer = keras.layers.ZeroPadding2D(padding=((1, 2), (2, 3)), data_format='channels_last')
        self._misc_conv_helper(layer, ishape)

    def test_embedding(self):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(1000, 64, input_length=10))
        input_array = np.random.randint(1000, size=(1, 10)).astype(np.float32)

        model.compile('rmsprop', 'mse')
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(input_array)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, input_array, expected))

    def _dot_helper(self, l2Normalize, input1, input2):
        data = [input1, input2]
        inputs = [keras.layers.Input(shape=d.shape[1:]) for d in data]

        layer = keras.layers.Dot(axes=-1, normalize=l2Normalize)(inputs)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_dot(self):
        self._dot_helper(False, self.asarray(1, 2, 3), self.asarray(4, 5, 6))
        self._dot_helper(True, self.asarray(1, 2, 3), self.asarray(4, 5, 6))

    def _batch_norm_helper(self, data, gamma, beta, scale, center, axis):
        model = keras.Sequential()
        layer = keras.layers.BatchNormalization(
            axis=axis,
            input_shape=data.shape[1:],
            moving_mean_initializer=keras.initializers.constant(np.mean(data)),
            moving_variance_initializer=keras.initializers.constant(np.var(data)),
            gamma_initializer=gamma,
            beta_initializer=beta,
            center=center,
            scale=scale,
        )
        model.add(layer)
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_batch_normalization(self):
        data = self.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        self._batch_norm_helper(data, 'ones', 'zeros', True, True, 1)
        self._batch_norm_helper(data, 'ones', 'zeros', True, True, 3)
        self._batch_norm_helper(data, 'ones', 'ones', True, True, 1)
        self._batch_norm_helper(data, 'ones', 'ones', True, True, 3)
        self._batch_norm_helper(data, 'ones', 'ones', True, False, 1)
        self._batch_norm_helper(data, 'zeros', 'zeros', False, True, 1)

    def test_simpleRNN(self):
        from keras.layers import SimpleRNN
        inputs1 = keras.Input(shape=(3, 1))
        cls = SimpleRNN(2, return_state=False, return_sequences=True)
        oname = cls(inputs1)  # , initial_state=t0)
        model = keras.Model(inputs=inputs1, outputs=[oname])
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_GRU(self):
        from keras.layers import GRU
        inputs1 = keras.Input(shape=(3, 1))
        cls = GRU(2, return_state=False, return_sequences=False)
        oname = cls(inputs1)
        model = keras.Model(inputs=inputs1, outputs=[oname])
        onnx_model = keras2onnx.convert_keras(model, model.name)

        data = np.array([0.1, 0.2, 0.3]).astype(np.float32).reshape((1, 3, 1))
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_LSTM(self):
        from keras.layers import LSTM
        inputs1 = keras.Input(shape=(3, 5))
        cls = LSTM(units=2, return_state=True, return_sequences=True)
        lstm1, state_h, state_c = cls(inputs1)
        model = keras.Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
        data = np.random.rand(3, 5).astype(np.float32).reshape((1, 3, 5))
        onnx_model = keras2onnx.convert_keras(model, model.name)

        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime(onnx_model.graph.name, onnx_model, data, expected))

    def test_LSTM_reshape(self):
        input_dim = 7
        sequence_len = 3
        inputs1 = keras.Input(shape=(sequence_len, input_dim))
        cls = keras.layers.LSTM(units=5, return_state=True, return_sequences=True)
        lstm1 = cls(inputs1)
        output = keras.layers.Reshape((sequence_len, 5))(lstm1[0])
        model = keras.Model(inputs=inputs1, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        onnx_model = keras2onnx.convert_keras(model, 'test')
        data = np.random.rand(input_dim, sequence_len).astype(np.float32).reshape((1, sequence_len, input_dim))
        expected = model.predict(data)
        self.assertTrue(self.run_onnx_runtime('tf_lstm', onnx_model, data, expected))

    def test_separable_convolution(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)
        model = keras.models.Sequential()
        model.add(keras.layers.SeparableConv2D(filters=10, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last', depth_multiplier=4))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, 'test')
        expected = model.predict(x)
        self.assertTrue(self.run_onnx_runtime('separable_convolution_1', onnx_model, x, expected))

        x = np.random.rand(N, H, C).astype(np.float32, copy=False)
        model = keras.models.Sequential()
        model.add(keras.layers.SeparableConv1D(filters=10, kernel_size=2, strides=1, padding='valid', input_shape=(H, C),
                         data_format='channels_last'))
        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, 'test')
        expected = model.predict(x)
        self.assertTrue(self.run_onnx_runtime('separable_convolution_2', onnx_model, x, expected))

    def test_recursive_model(self):
        from keras.layers import Input, Dense, Add

        N, C, D = 2, 3, 3
        x = np.random.rand(N, C).astype(np.float32, copy=False)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_mapped1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = Dense(D)(sub_input2)
        sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_mapped2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_2 = sub_model1(input1)
        mapped2_2 = sub_model2(input2)
        sub_sum = Add()([mapped1_2, mapped2_2])
        keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        x = [x, 2 * x]
        expected = keras_model.predict(x)
        self.assertTrue(self.run_onnx_runtime('recursive', onnx_model, x, expected))

    def test_recursive_and_shared_model(self):
        from keras.layers import Input, Dense, Add, Activation
        N, C, D = 2, 3, 3
        x = np.random.rand(N, C).astype(np.float32, copy=False)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_output1 = Activation('sigmoid')(sub_mapped1)
        sub_model1 = keras.Model(inputs=sub_input1, outputs=sub_output1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = sub_model1(sub_input2)
        sub_output2 = Activation('tanh')(sub_mapped2)
        sub_model2 = keras.Model(inputs=sub_input2, outputs=sub_output2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_1 = Activation('tanh')(input1)
        mapped2_1 = Activation('sigmoid')(input2)
        mapped1_2 = sub_model1(mapped1_1)
        mapped1_3 = sub_model1(mapped1_2)
        mapped2_2 = sub_model2(mapped2_1)
        sub_sum = Add()([mapped1_3, mapped2_2])
        keras_model = keras.Model(inputs=[input1, input2], outputs=sub_sum)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

        x = [x, 2 * x]
        expected = keras_model.predict(x)
        self.assertTrue(self.run_onnx_runtime('recursive_and_shared', onnx_model, x, expected))

    def test_channel_first_input(self):
        N, W, H, C = 2, 5, 6, 3
        inp1 = keras.layers.Input(batch_shape=(N, W, H, C), name='input1')
        inp2 = keras.layers.Input(batch_shape=(N, W, H, C), name='input2')
        output = keras.layers.Add()([inp1, inp2])
        model = keras.models.Model(inputs=[inp1, inp2], outputs=output)
        onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input1'])
        self.assertIsNotNone(onnx_model)

        data1 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
        data2 = np.random.rand(N, W, H, C).astype(np.float32).reshape((N, W, H, C))
        data_transpose = np.transpose(data1, (0, 3, 1, 2))
        self.assertTrue(data_transpose.shape == (N, C, W, H))

        expected = model.predict([data1, data2])
        self.assertTrue(self.run_onnx_runtime('channel_first_input', onnx_model, [data_transpose, data2], expected))

    def test_channel_last(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

        model.compile(optimizer='sgd', loss='mse')
        onnx_model = keras2onnx.convert_keras(model, channel_first_inputs=[model.inputs[0].name])

        expected = model.predict(x)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(onnx_model)

        x = np.transpose(x.astype(np.float32), [0, 3, 1, 2])
        self.assertTrue(self.run_onnx_runtime('channel_last_input', onnx_model, x, expected))

    def _test_keras_model(self, model, model_name='onnx_conversion', rtol=1.e-3, atol=1.e-5, img_size=224):
        img_path = os.path.join(os.path.dirname(__file__), 'data', 'elephant.jpg')
        try:
            img = image.load_img(img_path, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            onnx_model = keras2onnx.convert_keras(model, model.name)
            self.assertTrue(self.run_onnx_runtime(model_name, onnx_model, x, preds, rtol=rtol, atol=atol))
        except FileNotFoundError:
            self.assertTrue(False, 'The image data does not exist.')

    def test_MobileNet(self):
        from keras.applications import mobilenet
        model = mobilenet.MobileNet(weights='imagenet')
        model.save('mobile.h5')
        self._test_keras_model(model)

    @unittest.skipIf(StrictVersion(keras.__version__) < StrictVersion("2.2.3"),
                     "There is no mobilenet_v2 module before keras 2.2.3.")
    def test_MobileNetV2(self):
        from keras.applications import mobilenet_v2
        model = mobilenet_v2.MobileNetV2(weights='imagenet')
        self._test_keras_model(model)

    def on_Round(self, ctx, node, name, args):
        node.type = "Ceil"
        return node

    def on_StridedSlice(self, ctx, node, name, args):
        # node.type = "Reverse"
        # for now we implement common cases. Things like strides!=1 are not mappable to onnx.
        not_supported_attr = ["new_axis_mask"]
        for attr_name in not_supported_attr:
            attr = node.get_attr(attr_name)
            if attr is not None and attr.i != 0:
                raise ValueError("StridedSlice: attribute " + attr_name + " not supported")
        input_shape = ctx.get_shape(node.input[0])
        begin = node.inputs[1].get_tensor_value()
        end = node.inputs[2].get_tensor_value()
        strides = node.inputs[3].get_tensor_value()
        end_mask = node.get_attr("end_mask")
        end_mask = end_mask.i if end_mask is not None else 0
        ellipsis_mask = node.get_attr("ellipsis_mask")
        ellipsis_mask = ellipsis_mask.i if ellipsis_mask is not None else 0
        shrink_axis_mask = node.get_attr("shrink_axis_mask")
        shrink_axis_mask = shrink_axis_mask.i if shrink_axis_mask is not None else 0
        new_begin = []
        new_end = []
        axes = []
        # onnx slice op can't remove a axis, track axis and add a squeeze op if needed
        needs_squeeze = []
        import sys
        for idx, begin_item in enumerate(begin):
            end_item = end[idx]
            # if strides[idx] != 1:
            # raise ValueError("StridedSlice: only strides=1 is supported, current stride =" + str(strides[idx]))
            axes.append(idx)

            if (ellipsis_mask >> idx) & 1:
                new_begin.append(0)
                new_end.append(sys.maxsize)
                continue

            # an implicit condition is stride == 1 (checked in above)
            if begin_item < 0 and end_item == 0:
                end_item = sys.maxsize

            mask = (shrink_axis_mask >> idx) & 1
            if mask != 0:
                new_begin.append(begin_item)
                new_end.append(end_item)
                needs_squeeze.append(idx)
                continue

            new_begin.append(begin_item)
            mask = (end_mask >> idx) & 1
            if mask != 0:
                new_end.append(sys.maxsize)
            else:
                new_end.append(end_item)

        node.set_attr("starts", new_begin)
        node.set_attr("ends", new_end)
        node.set_attr("axes", axes)
        node.type = "Slice"
        ctx.remove_input(node, node.input[3])
        ctx.remove_input(node, node.input[2])
        ctx.remove_input(node, node.input[1])
        nodes = [node]
        from onnx import onnx_pb
        from tf2onnx import utils
        if needs_squeeze:
            name = utils.make_name(node.name)
            squeeze_node = ctx.insert_new_node_on_output("Squeeze", node.output[0], name)
            squeeze_node.set_attr("axes", needs_squeeze)
            nodes.append(squeeze_node)
            input_dtype = ctx.get_dtype(node.output[0])
            ctx.set_dtype(squeeze_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], squeeze_node.output[0])

        # onnx slice as of opset 7 does only take float tensors ... cast if needed
        input_dtype = ctx.get_dtype(node.input[0])
        if input_dtype != onnx_pb.TensorProto.FLOAT:
            if node.inputs[0].type == "Cast":
                # override the previous cast
                cast_node = node.inputs[0]
            else:
                cast_node = ctx.insert_new_node_on_input(node, "Cast", node.input[0])
                nodes.insert(0, cast_node)
            cast_node.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.FLOAT)
            ctx.copy_shape(node.input[0], cast_node.output[0])
            # undo the cast afer slice
            name = utils.make_name(node.name)
            cast_node = ctx.insert_new_node_on_output("Cast", nodes[-1].output[0], name)
            cast_node.set_attr("to", input_dtype)
            ctx.set_dtype(cast_node.output[0], input_dtype)
            ctx.copy_shape(node.output[0], cast_node.output[0])
            nodes.append(cast_node)
        return nodes

    def test_Concat(self):
        from keras import backend as K
        import tensorflow as tf
        from PIL import Image, ImageFont, ImageDraw
        image = Image.open('elephant.jpg')
        A, B, C, D, E = image.size[0], image.size[1], 10, 3, 2

        with tf.Session() as sess:
            box_xy = tf.placeholder(tf.float32, shape=(A, B, C, D, E))
            box_wh = tf.placeholder(tf.float32, shape=(A, B, C, D, E))
            input_shape = tf.placeholder(tf.float32, shape=(2,))
            image_shape = tf.placeholder(tf.float32, shape=(2,))
            box_yx = box_xy[..., ::-1]
            box_hw = box_wh[..., ::-1]
            input_shape = tf.cast(input_shape, K.dtype(box_yx))
            image_shape = tf.cast(image_shape, K.dtype(box_yx))
            new_shape = tf.round(image_shape * K.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)
            tensor = [
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2]  # x_max
            ]
            boxes = tf.concat(tensor, axis=-1)
            boxes *= tf.concat([image_shape, image_shape], axis=-1)
            boxes = K.reshape(boxes, [-1, 4])
            _ = tf.identity(boxes, name="output")
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, output_names=["output:0"], custom_op_handlers={'StridedSlice': (self.on_StridedSlice, []), 'Round': (self.on_Round, [])})
            model_proto = onnx_graph.make_model("test")
            with open("model.onnx", "wb") as f:
                f.write(model_proto.SerializeToString())

    def test_Concat2(self):
        from keras import backend as K
        import tensorflow as tf
        from PIL import Image, ImageFont, ImageDraw
        image = Image.open('elephant.jpg')
        A, B, C, D, E = image.size[0], image.size[1], 10, 3, 2

        with tf.Session() as sess:
            feats = tf.placeholder(tf.float32, shape=(A, B, C, 255))
            anchors = np.array([[116, 90], [156, 198], [373, 326]])
            num_anchors = len(anchors)
            # Reshape to batch, height, width, num_anchors, box_params.
            anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

            grid_shape = np.array([image.size[0], image.size[1]]) # K.shape(feats)[1:3]  # height, width
            grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                            [1, grid_shape[1], 1, 1])
            grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                            [grid_shape[0], 1, 1, 1])
            grid = K.concatenate([grid_x, grid_y])
            grid = K.cast(grid, K.dtype(feats))
            _ = tf.identity(grid, name="output")
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, output_names=["output:0"],
                                                         custom_op_handlers={'StridedSlice': (self.on_StridedSlice, []),
                                                                             'Round': (self.on_Round, [])})
            model_proto = onnx_graph.make_model("test")
            with open("model.onnx", "wb") as f:
                f.write(model_proto.SerializeToString())

    def test_dump(self):
        from onnx.onnx_pb import ModelProto
        m = ModelProto()
        s = None
        import json
        from google.protobuf.json_format import MessageToJson


        with open("yolov3.onnx", "rb") as f:
            m.ParseFromString(f.read())
            with open("new.json", 'w') as jsfile:
                actual_json_text = MessageToJson(m)
                jsfile.write(actual_json_text)



if __name__ == "__main__":
    unittest.main()

