import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import numpy as np


class EqDense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True, input_dim=None):
        super(EqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.input_dim = input_dim

    def build(self, input_shape):
        if self.input_dim == None:
            self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]), name=self.name + '_w')
            self.he_std = 2.0 / tf.sqrt(tf.cast(input_shape[-1], dtype='float32'))
        else:
            self.w = tf.Variable(tf.random.normal([self.input_dim, self.units]), name=self.name + '_w')
            self.he_std = 2.0 / tf.sqrt(tf.cast(self.input_dim, dtype='float32'))

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_vector = tf.matmul(inputs, self.w) * self.he_std
        if self.use_bias:
            return self.activation(feature_vector + self.b)
        else:
            return self.activation(feature_vector)


class Fir(kr.layers.Layer):
    def __init__(self, fir_filter, gain=1.0, upscale=False, downscale=False):
        super(Fir, self).__init__()
        self.fir_filter = fir_filter
        self.gain = gain
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, input_shape):
        fir_filter = tf.cast(self.fir_filter, dtype='float32')
        fir_filter = tf.tensordot(fir_filter, fir_filter, axes=0)
        fir_filter = fir_filter / tf.reduce_sum(fir_filter)
        fir_filter = fir_filter * self.gain
        if self.upscale:
            fir_filter = fir_filter * 4
            self.reshape_layer = kr.layers.Reshape([input_shape[1], input_shape[2] * 2, input_shape[3] * 2])
        self.fir_filter = tf.tile(fir_filter[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[1], 1])

    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            feature_maps = tf.stack([inputs, tf.zeros_like(inputs)], axis=3)
            feature_maps = tf.stack([feature_maps, tf.zeros_like(feature_maps)], axis=5)
            feature_maps = self.reshape_layer(feature_maps)
            return tf.nn.depthwise_conv2d(input=feature_maps, filter=self.fir_filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.fir_filter, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.fir_filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


class EqConv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True, downscale=False,
                 upscale=False, fir_filter=None):
        super(EqConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.fir_filter = fir_filter
        self.activation = activation
        self.use_bias = use_bias

        self.downscale = downscale
        self.upscale = upscale
        assert (downscale and upscale) != True
        if downscale or upscale:
            assert fir_filter != None

    def build(self, input_shape):
        input_filters = input_shape[1]

        self.he_std = 2.0 / tf.sqrt(tf.cast(self.kernel_size * self.kernel_size * input_filters, dtype='float32'))
        if self.upscale:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.filters, input_filters]),
                                 name=self.name + '_w')
            self.fir_layer = Fir(self.fir_filter, gain=self.he_std * 4)
        elif self.downscale:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                                 name=self.name + '_w')
            self.fir_layer = Fir(self.fir_filter, gain=self.he_std)
        else:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                                 name=self.name + '_w')

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.filters, 1, 1]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        input_shape = tf.shape(inputs)
        feature_maps = inputs

        if self.upscale:
            feature_maps = tf.nn.conv2d_transpose(input=feature_maps, filters=self.w,
                                                  output_shape=[input_shape[0], self.filters,
                                                                input_shape[2] * 2, input_shape[3] * 2],
                                                  strides=2, padding='SAME', data_format='NCHW')
            feature_maps = self.fir_layer(feature_maps)
        elif self.downscale:
            feature_maps = tf.nn.conv2d(self.fir_layer(inputs), self.w, strides=2, padding='SAME', data_format='NCHW')
        else:
            feature_maps = tf.nn.conv2d(feature_maps, self.w, strides=1, padding='SAME', data_format='NCHW') * self.he_std

        if self.use_bias:
            feature_maps = self.activation(feature_maps + self.b)
        else:
            feature_maps = self.activation(feature_maps)

        return feature_maps


class ModEqConv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True,
                 upscale=False, downscale=False, fir_filter=None):
        super(ModEqConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.upscale = upscale
        self.downscale = downscale
        self.fir_filter = fir_filter

        assert (upscale and downscale) != True
        if upscale or downscale:
            assert fir_filter != None

    def build(self, input_shape):
        input_filters = input_shape[1][1]
        self.scale_layer = EqDense(units=input_filters)

        self.he_std = 2.0 / tf.sqrt(tf.cast(self.kernel_size * self.kernel_size * input_filters, dtype='float32'))
        if self.upscale:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.filters, input_filters]),
                                 name=self.name + '_w')
            self.fir_layer = Fir(self.fir_filter, gain=self.he_std * 4)
        elif self.downscale:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                                 name=self.name + '_w')
            self.fir_layer = Fir(self.fir_filter, gain=self.he_std)
        else:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                                 name=self.name + '_w')

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.filters, 1, 1]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        style_vector = inputs[0]
        feature_maps = inputs[1]

        feature_maps_shape = tf.shape(feature_maps)

        scale_vector = self.scale_layer(style_vector)
        feature_maps = feature_maps * (scale_vector[:, :, tf.newaxis, tf.newaxis] + 1.0)

        if self.upscale:
            feature_maps = tf.nn.conv2d_transpose(input=feature_maps, filters=self.w,
                                                  output_shape=[feature_maps_shape[0], self.filters,
                                                                feature_maps_shape[2] * 2, feature_maps_shape[3] * 2],
                                                  strides=2, padding='SAME', data_format='NCHW')
            feature_maps = self.fir_layer(feature_maps)
        elif self.downscale:
            feature_maps = tf.nn.conv2d(self.fir_layer(feature_maps), self.w,
                                        strides=2, padding='SAME', data_format='NCHW')
        else:
            feature_maps = tf.nn.conv2d(feature_maps, self.w,
                                        strides=1, padding='SAME', data_format='NCHW') * self.he_std

        feature_maps = feature_maps / tf.math.reduce_std(feature_maps, axis=[2, 3], keepdims=True)

        if self.use_bias:
            feature_maps = self.activation(feature_maps + self.b)
        else:
            feature_maps = self.activation(feature_maps)

        return feature_maps


class ToRGB(kr.layers.Layer):
    def __init__(self):
        super(ToRGB, self).__init__()

    def build(self, input_shape):
        self.conv_layer = ModEqConv2D(filters=3, kernel_size=1)

    def call(self, inputs, *args, **kwargs):
        return self.conv_layer(inputs)


class FromRGB(kr.layers.Layer):
    def __init__(self, filters):
        super(FromRGB, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.conv_layer = EqConv2D(filters=self.filters, kernel_size=1, activation=tf.nn.leaky_relu)

    def call(self, inputs, *args, **kwargs):
        return self.conv_layer(inputs)


class TrainableNoise(kr.layers.Layer):
    def __init__(self, shape):
        super(TrainableNoise, self).__init__()
        self.shape = shape

    def build(self, input_shape):
        self.noise = tf.Variable(tf.random.normal(self.shape)[tf.newaxis], name=self.name + '_noise')

    def call(self, inputs, *args, **kwargs):
        return self.noise


class Mapper(kr.layers.Layer):
    def __init__(self):
        super(Mapper, self).__init__()

    def build(self, input_shape):
        style_vector = latent_vector = kr.Input([hp.latent_vector_dim])

        for _ in range(8):
            style_vector = EqDense(units=512, activation=tf.nn.leaky_relu)(style_vector)

        self.model = kr.Model(latent_vector, style_vector)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class Decoder(kr.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

    def build(self, input_shape):
        style_vector = kr.Input([512])
        feature_maps = TrainableNoise([512, 4, 4])(style_vector)

        feature_maps = ModEqConv2D(filters=512, kernel_size=3, activation=tf.nn.leaky_relu)([style_vector, feature_maps])
        fake_image = ToRGB()([style_vector, feature_maps])

        filters_sizes = [512, 512, 256, 128, 64]
        for filters in filters_sizes:
            feature_maps = ModEqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu,
                                       upscale=True, fir_filter=[1, 3, 3, 1])([style_vector, feature_maps])
            feature_maps = ModEqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)([style_vector, feature_maps])
            fake_image = Fir(fir_filter=[1, 3, 3, 1], upscale=True)(fake_image) + ToRGB()([style_vector, feature_maps])

        fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
        self.model = kr.Model(style_vector, fake_image)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class Discriminator(kr.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

    def build(self, input_shape):
        feature_maps = input_image = kr.Input(shape=[hp.image_resolution, hp.image_resolution, 3])
        feature_maps = tf.transpose(feature_maps, [0, 3, 1, 2])
        feature_maps = FromRGB(64)(feature_maps)

        filters_sizes = [64, 128, 256, 512, 512]
        for filters in filters_sizes:
            shortcut_feature_maps = EqConv2D(filters=np.minimum(filters * 2, 512), kernel_size=1, use_bias=False,
                                             downscale=True, fir_filter=[1, 3, 3, 1])(feature_maps)
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = EqConv2D(filters=np.minimum(filters * 2, 512), kernel_size=3, activation=tf.nn.leaky_relu,
                                    downscale=True, fir_filter=[1, 3, 3, 1])(feature_maps)
            feature_maps = (feature_maps + shortcut_feature_maps) / tf.sqrt(2.0)
        feature_maps = EqConv2D(filters=512, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
        feature_vector = kr.layers.Flatten()(feature_maps)
        feature_vector = EqDense(units=512, activation=tf.nn.leaky_relu, input_dim=512*4*4)(feature_vector)
        adv_value = EqDense(units=1)(feature_vector)
        latent_vector = EqDense(units=hp.latent_vector_dim)(feature_vector)
        self.model = kr.Model(input_image, [adv_value, latent_vector])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)
