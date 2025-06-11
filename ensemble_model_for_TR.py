import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input, TimeDistributed,  Conv1D, Conv2D, Dense, Add, MaxPool2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, GRU, LSTM, Dropout, BatchNormalization, Conv3D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import pickle
import asyncio


def add_coords_layer(input_tensor):
    batch_size_tensor = tf.shape(input_tensor)[0]
    x_dim = tf.shape(input_tensor)[1]
    y_dim = tf.shape(input_tensor)[2]

    xx_ones = tf.ones(tf.stack([batch_size_tensor, x_dim]), dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, axis=-1)

    xx_range = tf.tile(tf.expand_dims(
        tf.range(y_dim), axis=0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, axis=1)
    xx_channels = tf.matmul(xx_ones, xx_range)
    xx_channels = tf.expand_dims(xx_channels, axis=-1)

    yy_ones = tf.ones(tf.stack([batch_size_tensor, y_dim]), dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, axis=-1)

    yy_range = tf.tile(tf.expand_dims(
        tf.range(x_dim), axis=0), [batch_size_tensor, 1])
    yy_range = tf.expand_dims(yy_range, axis=1)
    yy_channels = tf.matmul(yy_ones, yy_range)
    yy_channels = tf.expand_dims(yy_channels, axis=-1)

    x_dim = tf.cast(x_dim, tf.float32)
    y_dim = tf.cast(y_dim, tf.float32)

    xx_channels = tf.cast(xx_channels, tf.float32)/(y_dim-1)
    xx_channels = (xx_channels * 2) - 1

    yy_channels = tf.cast(yy_channels, tf.float32)/(x_dim-1)
    yy_channels = (yy_channels * 2) - 1

    outputs = tf.concat([input_tensor, xx_channels, yy_channels], axis=-1)

    return outputs


def bottleneck_mobile(bottom, num_out_channels, block_name, activation_function='relu'):

    # Skip layer

    if K.int_shape(bottom)[-1] == num_out_channels:
        skip = bottom
    else:
        skip = Conv2D(num_out_channels, kernel_size=(
            1, 1), activation=activation_function, padding='same', name=block_name + 'skip')(bottom)

    # Residual layer : 3 convolutional blocks, [n_channels_out/2, n_channels_out/2 n_channels_out]
    x = tf.keras.layers.Activation(activation_function)(bottom)
    x = BatchNormalization(name=block_name + 'batch_1')(x)
    x = Conv2D(num_out_channels/2, kernel_size=(1, 1),
               padding='same', name=block_name + 'conv_1x1_x1')(x)

    x1 = tf.keras.layers.Activation(activation_function)(x)
    x1 = BatchNormalization(name=block_name + 'batch_2')(x1)
    x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), dilation_rate=(
        1, 1), padding='same', name=block_name + 'conv_3x3_x1')(x1)

    x2 = tf.keras.layers.Activation(activation_function)(x)
    x2 = BatchNormalization(name=block_name + 'batch_3')(x2)
    x2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), dilation_rate=(
        2, 2), padding='same', name=block_name + 'conv_5x5_x1')(x2)

    x3 = tf.keras.layers.Activation(activation_function)(x)
    x3 = BatchNormalization(name=block_name + 'batch_4')(x3)
    x3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), dilation_rate=(
        3, 3), padding='same',  name=block_name + 'conv_7x7_x1')(x3)

    x_final = Add()([x1, x2, x3])

    x_out = tf.keras.layers.Activation(activation_function)(x_final)
    x_out = BatchNormalization(name=block_name + 'batch_5')(x_out)
    x_out = Conv2D(num_out_channels, kernel_size=(1, 1), activation=activation_function,
                   padding='same', name=block_name + 'conv_3x3_x2')(x_out)

    x_out = Add(name=block_name + '_residual')([skip, x_out])

    # x_out = tf.keras.layers.Activation('sigmoid', name = block_name + '_sigmoid_output')(x_out)

    return x_out


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels, left_features):
    """
    Create the half blocks for the hourglass module with 1/2, 1/4, 1/8 resolutions
    """
    hgname = 'hg' + str(hglayer)

    if left_features:
        lf1, lf2, lf3, lf4 = left_features

        f1 = bottleneck(bottom, num_channels, hgname + '_l1')  # 256256258
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)
        f2 = bottleneck(Add()([x, lf2]), num_channels, hgname + '_l2')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)
        f4 = bottleneck(Add()([x, lf3]), num_channels, hgname + '_l4')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)
        f8 = bottleneck(Add()([x, lf4]), num_channels, hgname + '_l8')

    else:
        f1 = bottleneck(bottom, num_channels, hgname + '_l1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)
        f2 = bottleneck(x, num_channels, hgname + '_l2')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)
        f4 = bottleneck(x, num_channels, hgname + '_l4')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)
        f8 = bottleneck(x, num_channels, hgname + '_l8')

    return f1, f2, f4, f8


def connect_left_to_right(left, right, bottleneck, name, num_channels):
    x_left = bottleneck(left, num_channels, name + '_connect')
    x_right = UpSampling2D()(right)
    add = Add()([x_left, x_right])
    out = bottleneck(add, num_channels, name + '_connect_conv')

    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    lf8_connect = bottleneck(lf8, num_channels, str(hgid) + '_lf8')
    x = bottleneck(lf8,  num_channels, str(hgid) + 'lf8_x1')  # 32,32,64
    x = bottleneck(x, num_channels, str(hgid) + '_lf8_x2')
    x = bottleneck(x, num_channels, str(hgid) + '_lf8_x3')

    rf8 = Add()([x, lf8_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    """
    Apply the left to right bottleneck to each of the left features to get the right features.
    """
    lf1, lf2, lf4, lf8 = leftfeatures
    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)
    rf4 = connect_left_to_right(
        lf4, rf8, bottleneck, 'hg' + str(hglayer) + '_rf4', num_channels)
    rf2 = connect_left_to_right(
        lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels)
    rf1 = connect_left_to_right(
        lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1_out_256_256', num_channels)

    return rf1


def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    """
    Head to next stage + Head to intermediate features.
    """

    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu',
                  padding='same', name=str(hgid) + '_conv1x1_x1')(rf1)
    head = BatchNormalization(name=str(hgid) + 'bn')(head)
    # num_class
    # head_parts = Conv2D(num_classes, kernel_size = (1, 1), activation = 'linear', padding = 'same', name = str(hgid) + 'conv_1x1_parts')(head)
    head_parts = Conv2D(num_classes, kernel_size=(
        1, 1), activation='softmax', padding='same', name=str(hgid) + 'conv_1x1_parts')(head)
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                  padding='same', name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                    padding='same', name=str(hgid) + '_conv_1x1_x3')(head_parts)
    head_next_stage = Add()([head, head_m, prelayerfeatures])

    # 這裡試著加加看sigmoid layer
    # head_next_stage = tf.keras.layers.Conv2D(3,kernel_size=(1,1), activation = 'linear', padding = 'same', name = str(hgid) + '_output')(head_next_stage)

    return head_next_stage, head_parts


def create_heads_for_pose(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    """
    Head to next stage + Head to intermediate features.
    """

    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu',
                  padding='same', name=str(hgid) + '_conv1x1_x1')(rf1)
    head = BatchNormalization(name=str(hgid) + 'bn')(head)
    # num_class
    head_parts = Conv2D(num_classes, kernel_size=(
        1, 1), activation='linear', padding='same', name=str(hgid) + 'conv_1x1_parts')(head)
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                  padding='same', name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                    padding='same', name=str(hgid) + '_conv_1x1_x3')(head_parts)
    head_next_stage = Add()([head, head_m, prelayerfeatures])

    # 這裡試著加加看sigmoid layer
    # head_next_stage = tf.keras.layers.Conv2D(3,kernel_size=(1,1), activation = 'linear', padding = 'same', name = str(hgid) + '_output')(head_next_stage)

    return head_next_stage, head_parts


def hourglass_module(bottom, num_classes, num_channels, bottleneck, hgid, left_features):
    bottom = tf.keras.layers.Conv2D(
        num_channels, (1, 1), padding='same')(bottom)  # 3->64

    bottom_expand = add_coords_layer(bottom)

    left_features = create_left_half_blocks(
        bottom_expand, bottleneck, hgid, num_channels, left_features)

    rf1 = create_right_half_blocks(
        left_features, bottleneck, hgid, num_channels)

    head_next_stage, head_parts = create_heads(
        bottom, rf1, num_classes, hgid, num_channels)

    return head_next_stage, head_parts, left_features


def hourglass_module_for_pose(bottom, num_classes, num_channels, bottleneck, hgid, left_features):
    bottom = tf.keras.layers.Conv2D(
        num_channels, (1, 1), padding='same')(bottom)  # 3->64

    bottom_expand = add_coords_layer(bottom)

    left_features = create_left_half_blocks(
        bottom_expand, bottleneck, hgid, num_channels, left_features)

    rf1 = create_right_half_blocks(
        left_features, bottleneck, hgid, num_channels)

    head_next_stage, head_parts = create_heads(
        bottom, rf1, num_classes, hgid, num_channels)

    return head_next_stage, head_parts, left_features


def create_front_module(input, num_channels, bottleneck):
    # front module, input to 1/4 of the resolution.
    # 1 (7, 7) conv2D + maxpool + 3 residual layers.
    # 加上 Cood_Conv layer
    x = add_coords_layer(input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same',
               activation='relu', name='front_conv_1x1_x1')(x)
    x = BatchNormalization(name='front')(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x1')
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x2')
    x = bottleneck(x, num_channels, 'front_residual_x3')

    return x


async def create_hourglass_network(num_classes, num_stacks, num_channels, inres, outres, bottleneck):
    input = Input(shape=(*inres[:2], 3))

    front_features = create_front_module(input, num_channels, bottleneck)

    head_next_stage = front_features

    left_features = None

    outputs = []
    head_next_stage_lst = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss, left_features = hourglass_module(
            head_next_stage, num_classes, num_channels, bottleneck, i, left_features)
        outputs.append(head_to_loss)
        head_next_stage_lst.append(head_next_stage)

    model = tf.keras.Model(inputs=input, outputs=outputs)

    return model


def create_hourglass_network_for_pose(num_classes, num_stacks, num_channels, inres, outres, bottleneck):
    input = Input(shape=(*inres[:2], 3))

    front_features = create_front_module(input, num_channels, bottleneck)

    head_next_stage = front_features

    left_features = None

    outputs = []
    head_next_stage_lst = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss, left_features = hourglass_module_for_pose(
            head_next_stage, num_classes, num_channels, bottleneck, i, left_features)
        outputs.append(head_to_loss)
        head_next_stage_lst.append(head_next_stage)

    model = tf.keras.Model(inputs=input, outputs=outputs)

    return model


class Conv3dLeaky(tf.keras.layers.Layer):
    def __init__(self, filters, kernels, strides):
        super(Conv3dLeaky, self).__init__()
        self.conv3d = tf.keras.layers.Conv3D(
            filters, kernels, strides, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inp):
        x = self.conv3d(inp)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Conv3dLinear(tf.keras.layers.Layer):
    def __init__(self, filters, kernels, strides):
        super(Conv3dLinear, self).__init__()
        self.conv3d = tf.keras.layers.Conv3D(
            filters, kernels, strides, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inp):
        x = self.conv3d(inp)
        x = self.bn(x)

        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv3dLeaky(channels, 1, 1)
        self.conv2 = Conv3dLeaky(channels, 3, 1)
        self.conv3 = Conv3dLinear(channels, 1, 1)  # linear
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, inp):
        x_0 = self.conv1(inp)  # 128
        x_1 = self.conv2(x_0)  # 128
        temp = self.conv3(x_1)
        temp = self.bn(temp)
        out = tf.keras.layers.Add()([temp, x_0])
        out = self.activation(out)
        return out


class CspResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, two_residual_block=True, **kwargs):
        super(CspResBlock, self).__init__(**kwargs)

        self.two_residual_block = two_residual_block
        self.conv_for_csp = tf.keras.layers.Conv3D(
            64, 1, 1, padding='same', activation='linear')
        if two_residual_block is True:
            self.residual_block_0 = ResidualBlock(channels)
            self.residual_block_1 = ResidualBlock(channels)
            self.conv_post_residual = Conv3dLeaky(64, 1, 1)

            self.conv_for_concat = tf.keras.layers.Conv3D(
                64, 1, 1, padding='same')

        if two_residual_block is False:
            self.residual_block_0 = ResidualBlock(channels)

            self.conv_post_residual = Conv3dLeaky(64, 1, 1)
            self.conv_for_concat = tf.keras.layers.Conv3D(
                64, 1, 1, padding='same')

    def call(self, inp):
        csp_shortcut, original = tf.split(inp, [int(
            inp.shape[-1]*0.5), inp.shape[-1]-int(inp.shape[-1]*0.5)], -1)  # 256,256,10,3
        # 256,256,10,128 # linear output
        shortcut = self.conv_for_csp(csp_shortcut)

        # put two residual block in one csp net block
        if self.two_residual_block is True:
            original = self.residual_block_0(original)
            original = self.residual_block_1(original)
            original = self.conv_post_residual(original)

            concat = tf.keras.layers.concatenate([shortcut, original], axis=-1)

        if self.two_residual_block is False:
            original = self.residual_block_0(original)

            original = self.conv_post_residual(original)

            concat = tf.keras.layers.concatenate([shortcut, original], axis=-1)

        return concat


class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, reshape_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.reshape_shape = reshape_shape

    def call(self, x):
        u = tf.reshape(x, self.reshape_shape)

        return u


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, transpose_shape, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.transpose_shape = transpose_shape

    def call(self, x):
        u = tf.transpose(x, self.transpose_shape)

        return u


async def cnn_lstm():  # 70%
    inp = Input(shape=(20, 128, 128, 3))
    x = Conv3D(8, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Conv3D(16, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv3D(64, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv3D(128, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    # x = Conv3D(256,(3,3,3),strides = (1,2,2),padding = 'same', activation = 'relu')(x)
    # x = Dropout(0.3)(x)
    x = ReshapeLayer([-1, 20, 1024])(x)

    x = LSTM(32, return_sequences=True, dropout=0.3)(x)  # return 32
    x = LSTM(32, return_sequences=True, dropout=0.3)(
        x)  # return 32 per time step (None, 20, 32)
    out = Conv1D(2, 3, padding='same', activation='sigmoid')(x)  # None, 20, 2
    mod = models.Model(inputs=inp, outputs=out)
    return mod


epsilon = 1e-7


def squash(s):
    s_norm = tf.norm(s, axis=-1, keepdims=True)
    return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)


class CapsuleLayer_for_final_model(tf.keras.layers.Layer):

    def __init__(self, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer_for_final_model, self).__init__(**kwargs)

        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

        with tf.name_scope("Variables") as scope:
            self.W = tf.Variable(tf.random_normal_initializer()(shape=[
                                 1, 8*8*8, 2, 16, 16]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            # self.W = self.add_weight(shape=[1, 16*16*8, 2, 16, 16],initializer=self.kernel_initializer,name='W')

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule, 1]
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(inputs, (-1,  8*8*8, 16))
            u = tf.expand_dims(u, axis=-2)
            u = tf.expand_dims(u, axis=-1)  # u.shape: (None, 1152, 1, 8, 1)
            # In the matrix multiplication: (1, 1752, 29, 4, 8) x (None, 1752, 1, 8, 1) -> (None, 1752, 29, 4, 1)
            u_hat = tf.matmul(self.W, u)  # u_hat.shape: (None, 1752, 29, 4, 1)
            u_hat = tf.squeeze(u_hat, [4])

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, 1, self.input_num_capsule].
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros(shape=[tf.shape(inputs)[0],  8*8*8, 2, 1])

            assert self.routings > 0, 'The routings should be > 0.'
            for i in range(self.routings):
                # c.shape=[batch_size, num_capsule, 1, input_num_capsule]
                c = tf.nn.softmax(b, axis=-2)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
                v = squash(s)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(
                    u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4])
                b += agreement

        return v

    def get_config(self):
        config = {
            'routings': self.routings
        }
        base_config = super(CapsuleLayer_for_final_model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvAttentionLayer, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='tanh')

    def call(self, x):
        x_gap = tf.keras.layers.GlobalAveragePooling3D()(x)  # None, 1,1,128
        x_gap = tf.reshape(x_gap, [-1, 128])
        x_gap = self.dense_1(x_gap)
        x_gap = self.dense_2(x_gap)
        x_gap = tf.reshape(x_gap, [-1, 1, 1, 1, 128])
        x_temp = tf.multiply(x, x_gap)

        out = tf.keras.layers.Add()([x, x_temp])

        return out


class CapsAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CapsAttentionLayer, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='tanh')

    def call(self, x):
        x_gap = tf.reshape(x, [-1, 8*8*128])

        x_gap = self.dense_1(x_gap)
        x_gap = self.dense_2(x_gap)
        x_gap = tf.reshape(x_gap, [-1, 1, 1, 128])
        x_temp = tf.multiply(x, x_gap)  # 6,6,128  / 1,1,128

        out = tf.keras.layers.Add()([x, x_temp])

        return out


# 加上時間維度的embedding
def add_coords_layer_frame(input_tensor):
    ''' input tensor = None,20,256*256,3'''

    batch_size_tensor = tf.shape(input_tensor)[0]
    x_dim = tf.shape(input_tensor)[1]  # 20
    y_dim = tf.shape(input_tensor)[2]  # 256*256

    xx_ones = tf.ones(tf.stack([batch_size_tensor, y_dim]), dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, axis=1)

    xx_range = tf.tile(tf.expand_dims(
        tf.range(x_dim), axis=0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, axis=-1)  # 1,20,1
    xx_channels = tf.matmul(xx_range, xx_ones)
    xx_channels = tf.expand_dims(xx_channels, axis=-1)

    x_dim = tf.cast(x_dim, tf.float32)
    y_dim = tf.cast(y_dim, tf.float32)

    xx_channels = tf.cast(xx_channels, tf.float32)/(x_dim-1)
    xx_channels = (xx_channels * 2) - 1
    outputs = tf.concat([input_tensor, xx_channels], axis=-1)

    return outputs


class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, reshape_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.reshape_shape = reshape_shape

    def call(self, x):
        u = tf.reshape(x, self.reshape_shape)

        return u


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, transpose_shape, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.transpose_shape = transpose_shape

    def call(self, x):
        u = tf.transpose(x, self.transpose_shape)

        return u


class SegLayerRegur(tf.keras.layers.Layer):
    def __init__(self, model, out_shape, **kwargs):
        super(SegLayerRegur, self).__init__(**kwargs)

        self.model = model
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        return self.model(inputs)[:, :, :, 0]  # 不需要background，0 for regurg,

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, self.out_shape)


class SegLayerChamber(tf.keras.layers.Layer):
    def __init__(self, model, out_shape, **kwargs):
        super(SegLayerChamber, self).__init__(**kwargs)

        self.model = model
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        return self.model(inputs)[:, :, :, 0]  # 0 for atrium

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, self.out_shape)


class SegLayerChamberA4C(tf.keras.layers.Layer):
    def __init__(self, model, out_shape, **kwargs):
        super(SegLayerChamberA4C, self).__init__(**kwargs)

        self.model = model
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        return self.model(inputs)[:, :, :, 2]  # 因為A4C的第3個channel才是RA

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, self.out_shape)


# merge seg and systole/diastole layer
class SystoleSeg(tf.keras.layers.Layer):
    def __init__(self, seg_mod, chamber_seg, event_detector_mod, out_shape, **kwargs):
        super(SystoleSeg, self).__init__(**kwargs)
        self.seg_mod = seg_mod
        self.chamber_seg = chamber_seg
        self.event_mod = event_detector_mod
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        systole_output = self.event_mod(inputs)
        systole_output = tf.argmax(systole_output, -1)
        systole_output = tf.expand_dims(systole_output, -1)
        systole_output = tf.cast(tf.reshape(
            systole_output, [-1, 20, 1, 1, 1]), dtype='float32')  # 1,0,0
        # diastole_output = tf.math.subtract(1.0,systole_output) # 0,1,1,
        # systole_diastole_concat = tf.concat([systole_output, diastole_output], -1)

        chamber_output = tf.keras.layers.TimeDistributed(
            SegLayerChamber(self.chamber_seg, 1))(inputs)
        # channel 0 for left atrium, channel 1 for left ventricle
        chamber_output_ = tf.where(chamber_output > 0.5, 1.0, 0.0)

        color_output = tf.keras.layers.TimeDistributed(SegLayerRegur(self.seg_mod, 1))(
            inputs)  # channel 0 : regur / chnnel 1 : inflow

        color_output_ = tf.where(color_output > 0.5, 1.0, 0.0)

        out = tf.multiply(systole_output, color_output_)  # None, 20, 256,256,2

        out_ = tf.multiply(chamber_output_, out)

        return out_

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, self.out_shape)  # 20,256,256,1


# merge seg and systole/diastole layer
class SystoleSegA4C(tf.keras.layers.Layer):
    def __init__(self, seg_mod, chamber_seg, event_detector_mod, out_shape, **kwargs):
        super(SystoleSegA4C, self).__init__(**kwargs)
        self.seg_mod = seg_mod
        self.chamber_seg = chamber_seg
        self.event_mod = event_detector_mod
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        systole_output = self.event_mod(inputs)
        systole_output = tf.argmax(systole_output, -1)
        systole_output = tf.expand_dims(systole_output, -1)
        systole_output = tf.cast(tf.reshape(
            systole_output, [-1, 20, 1, 1, 1]), dtype='float32')  # 1,0,0
        # diastole_output = tf.math.subtract(1.0,systole_output) # 0,1,1,
        # systole_diastole_concat = tf.concat([systole_output, diastole_output], -1)

        chamber_output = tf.keras.layers.TimeDistributed(
            SegLayerChamberA4C(self.chamber_seg, 1))(inputs)
        # channel 0 for left atrium, channel 1 for left ventricle
        chamber_output_ = tf.where(chamber_output > 0.5, 1.0, 0.0)

        color_output = tf.keras.layers.TimeDistributed(SegLayerRegur(self.seg_mod, 1))(
            inputs)  # channel 0 : regur / chnnel 1 : inflow

        color_output_ = tf.where(color_output > 0.5, 1.0, 0.0)

        out = tf.multiply(systole_output, color_output_)  # None, 20, 256,256,2

        out_ = tf.multiply(chamber_output_, out)

        return out_

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 128, 128, self.out_shape)  # 20,256,256,1


channels = 32
two_residual_block = False


def ensemble_model(output_bias, channels, two_residual_block, event_detector_mod, color_seg, RV_chamber_seg, SAX_chamber_seg, a4c_chamber_seg):
    inp_RV = Input((20, 128, 128, 3), name="RV")
    inp_SAX = Input((20, 128, 128, 3), name="SAX")
    inp_a4c = Input((20, 128, 128, 3), name="a4c")
    inp_RV_color = Input((20, 128, 128, 3), name="RV_color")
    inp_SAX_color = Input((20, 128, 128, 3), name="SAX_color")
    inp_a4c_color = Input((20, 128, 128, 3), name="a4c_color")

    #####
    inp_RV_time_embed = ReshapeLayer((-1, 20, 128*128, 3))(inp_RV)
    inp_RV_time_embed = add_coords_layer_frame(inp_RV_time_embed)
    inp_RV_time_embed = ReshapeLayer((-1, 20, 128, 128, 4))(inp_RV_time_embed)

    RV_raw_embed_merge = TransposeLayer((0, 3, 2, 1, 4))(inp_RV_time_embed)
    RV_raw_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        RV_raw_embed_merge)  # 256,256,20,16

    RV_raw_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_RV_raw')(RV_raw_embed_merge)

    RV_raw_temp = tf.keras.layers.LeakyReLU()(RV_raw_temp)

    RV_raw_temp = ConvAttentionLayer()(RV_raw_temp)

    RV_raw_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_RV_raw')(
        RV_raw_temp)  # 128,128,10,128
    RV_raw_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_raw_temp_1)  # 64,64,5,128

    RV_raw_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_RV_raw')(
        RV_raw_temp_1)  # 64,64,5,128
    RV_raw_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_raw_temp_2)  # 32,32,2,128

    RV_raw_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_RV_raw')(
        RV_raw_temp_2)  # 32,32,2,128
    RV_raw_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_raw_temp_3)  # 8,8,1,128

    RV_raw_reshape = tf.reshape(RV_raw_temp_3, [-1, 8, 8, 128])

    RV_raw_reshape = CapsAttentionLayer()(RV_raw_reshape)
    v_for_RV_raw = CapsuleLayer_for_final_model(
        name='capsule_output_for_RV_raw')(RV_raw_reshape)  # 1,2,16
    ####

    RV_color_systole_seg_out = SystoleSeg(
        color_seg, RV_chamber_seg, event_detector_mod, 1, name="RV_color_systole_output")(inp_RV_color)
    RV_color_merge = tf.keras.layers.Concatenate(
        axis=-1)([inp_RV_color, RV_color_systole_seg_out])  # 20,256,256,15

    inp_RV_color_time_embed = ReshapeLayer(
        (-1, 20, 128*128, 4))(RV_color_merge)
    inp_RV_color_time_embed = add_coords_layer_frame(inp_RV_color_time_embed)
    inp_RV_color_time_embed = ReshapeLayer(
        (-1, 20, 128, 128, 5))(inp_RV_color_time_embed)

    RV_color_embed_merge = TransposeLayer(
        (0, 3, 2, 1, 4))(inp_RV_color_time_embed)
    RV_color_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        RV_color_embed_merge)  # 20,256,256,5

    RV_color_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_RV_color')(RV_color_embed_merge)
    RV_color_temp = tf.keras.layers.LeakyReLU()(RV_color_temp)
    RV_color_temp = ConvAttentionLayer()(RV_color_temp)

    RV_color_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_RV_color')(
        RV_color_temp)  # 128,128,10,128
    RV_color_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_color_temp_1)  # 64,64,5,128

    RV_color_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_RV_color')(
        RV_color_temp_1)  # 64,64,5,128
    RV_color_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_color_temp_2)  # 32,32,2,128

    RV_color_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_RV_color')(
        RV_color_temp_2)  # 32,32,2,128
    RV_color_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(RV_color_temp_3)  # 16,16,1,128

    RV_color_reshape = tf.reshape(RV_color_temp_3, [-1, 8, 8, 128])

    RV_color_reshape = CapsAttentionLayer()(RV_color_reshape)
    v_for_RV_color = CapsuleLayer_for_final_model(
        name='capsule_output_for_RV_color')(RV_color_reshape)  # 1,2,16

    ###
    # SAX

    inp_SAX_time_embed = ReshapeLayer((-1, 20, 128*128, 3))(inp_SAX)
    inp_SAX_time_embed = add_coords_layer_frame(inp_SAX_time_embed)
    inp_SAX_time_embed = ReshapeLayer(
        (-1, 20, 128, 128, 4))(inp_SAX_time_embed)

    SAX_raw_embed_merge = TransposeLayer((0, 3, 2, 1, 4))(inp_SAX_time_embed)
    SAX_raw_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        SAX_raw_embed_merge)  # 256,256,20,16

    SAX_raw_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_SAX_raw')(SAX_raw_embed_merge)
    SAX_raw_temp = tf.keras.layers.LeakyReLU()(SAX_raw_temp)
    SAX_raw_temp = ConvAttentionLayer()(SAX_raw_temp)
    SAX_raw_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_SAX_raw')(
        SAX_raw_temp)  # 128,128,10,128
    SAX_raw_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_raw_temp_1)  # 64,64,5,128

    SAX_raw_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_SAX_raw')(
        SAX_raw_temp_1)  # 64,64,5,128
    SAX_raw_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_raw_temp_2)  # 32,32,2,128

    SAX_raw_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_SAX_raw')(
        SAX_raw_temp_2)  # 32,32,2,128
    SAX_raw_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_raw_temp_3)  # 8,8,1,128

    SAX_raw_reshape = tf.reshape(SAX_raw_temp_3, [-1, 8, 8, 128])

    SAX_raw_reshape = CapsAttentionLayer()(SAX_raw_reshape)
    v_for_SAX_raw = CapsuleLayer_for_final_model(
        name='capsule_output_for_SAX_raw')(SAX_raw_reshape)  # 1,2,16
    ####

    SAX_color_systole_seg_out = SystoleSeg(
        color_seg, SAX_chamber_seg, event_detector_mod, 1, name="SAX_color_systole_output")(inp_SAX_color)
    SAX_color_merge = tf.keras.layers.Concatenate(
        axis=-1)([inp_SAX_color, SAX_color_systole_seg_out])  # 20,256,256,15

    inp_SAX_color_time_embed = ReshapeLayer(
        (-1, 20, 128*128, 4))(SAX_color_merge)
    inp_SAX_color_time_embed = add_coords_layer_frame(inp_SAX_color_time_embed)
    inp_SAX_color_time_embed = ReshapeLayer(
        (-1, 20, 128, 128, 5))(inp_SAX_color_time_embed)

    SAX_color_embed_merge = TransposeLayer(
        (0, 3, 2, 1, 4))(inp_SAX_color_time_embed)
    SAX_color_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        SAX_color_embed_merge)  # 20,256,256,5

    SAX_color_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_SAX_color')(SAX_color_embed_merge)
    SAX_color_temp = tf.keras.layers.LeakyReLU()(SAX_color_temp)
    SAX_color_temp = ConvAttentionLayer()(SAX_color_temp)
    SAX_color_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_SAX_color')(
        SAX_color_temp)  # 128,128,10,128
    SAX_color_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_color_temp_1)  # 64,64,5,128

    SAX_color_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_SAX_color')(
        SAX_color_temp_1)  # 64,64,5,128
    SAX_color_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_color_temp_2)  # 32,32,2,128

    SAX_color_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_SAX_color')(
        SAX_color_temp_2)  # 32,32,2,128
    SAX_color_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(SAX_color_temp_3)  # 16,16,1,128

    SAX_color_reshape = tf.reshape(SAX_color_temp_3, [-1, 8, 8, 128])

    SAX_color_reshape = CapsAttentionLayer()(SAX_color_reshape)
    v_for_SAX_color = CapsuleLayer_for_final_model(
        name='capsule_output_for_SAX_color')(SAX_color_reshape)  # 1,2,16

    ###
    # a4c
    inp_a4c_time_embed = ReshapeLayer((-1, 20, 128*128, 3))(inp_a4c)
    inp_a4c_time_embed = add_coords_layer_frame(inp_a4c_time_embed)
    inp_a4c_time_embed = ReshapeLayer(
        (-1, 20, 128, 128, 4))(inp_a4c_time_embed)

    a4c_raw_embed_merge = TransposeLayer((0, 3, 2, 1, 4))(inp_a4c_time_embed)
    a4c_raw_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        a4c_raw_embed_merge)  # 256,256,20,16

    a4c_raw_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_a4c_raw')(a4c_raw_embed_merge)
    a4c_raw_temp = tf.keras.layers.LeakyReLU()(a4c_raw_temp)
    a4c_raw_temp = ConvAttentionLayer()(a4c_raw_temp)
    a4c_raw_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_a4c_raw')(
        a4c_raw_temp)  # 128,128,10,128
    a4c_raw_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_raw_temp_1)  # 64,64,5,128

    a4c_raw_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_a4c_raw')(
        a4c_raw_temp_1)  # 64,64,5,128
    a4c_raw_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_raw_temp_2)  # 32,32,2,128

    a4c_raw_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_a4c_raw')(
        a4c_raw_temp_2)  # 32,32,2,128
    a4c_raw_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_raw_temp_3)  # 8,8,1,128

    a4c_raw_reshape = tf.reshape(a4c_raw_temp_3, [-1, 8, 8, 128])

    a4c_raw_reshape = CapsAttentionLayer()(a4c_raw_reshape)
    v_for_a4c_raw = CapsuleLayer_for_final_model(
        name='capsule_output_for_a4c_raw')(a4c_raw_reshape)  # 1,2,16
    ####

    a4c_color_systole_seg_out = SystoleSegA4C(
        color_seg, a4c_chamber_seg, event_detector_mod, 1, name="a4c_color_systole_output")(inp_a4c_color)
    a4c_color_merge = tf.keras.layers.Concatenate(
        axis=-1)([inp_a4c_color, a4c_color_systole_seg_out])  # 20,256,256,15

    inp_a4c_color_time_embed = ReshapeLayer(
        (-1, 20, 128*128, 4))(a4c_color_merge)
    inp_a4c_color_time_embed = add_coords_layer_frame(inp_a4c_color_time_embed)
    inp_a4c_color_time_embed = ReshapeLayer(
        (-1, 20, 128, 128, 5))(inp_a4c_color_time_embed)

    a4c_color_embed_merge = TransposeLayer(
        (0, 3, 2, 1, 4))(inp_a4c_color_time_embed)
    a4c_color_embed_merge = TransposeLayer((0, 2, 1, 3, 4))(
        a4c_color_embed_merge)  # 20,256,256,5

    a4c_color_temp = tf.keras.layers.Conv3D(
        128, 3, strides=2, padding='same', name='conv_0_a4c_color')(a4c_color_embed_merge)
    a4c_color_temp = tf.keras.layers.LeakyReLU()(a4c_color_temp)
    a4c_color_temp = ConvAttentionLayer()(a4c_color_temp)
    a4c_color_temp_1 = CspResBlock(channels, two_residual_block, name='csp_0_a4c_color')(
        a4c_color_temp)  # 128,128,10,128
    a4c_color_temp_1 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_color_temp_1)  # 64,64,5,128

    a4c_color_temp_2 = CspResBlock(channels, two_residual_block, name='csp_1_a4c_color')(
        a4c_color_temp_1)  # 64,64,5,128
    a4c_color_temp_2 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_color_temp_2)  # 32,32,2,128

    a4c_color_temp_3 = CspResBlock(channels, two_residual_block, name='csp_2_a4c_color')(
        a4c_color_temp_2)  # 32,32,2,128
    a4c_color_temp_3 = tf.keras.layers.MaxPooling3D(
        (2, 2, 2))(a4c_color_temp_3)  # 16,16,1,128

    a4c_color_reshape = tf.reshape(a4c_color_temp_3, [-1, 8, 8, 128])

    a4c_color_reshape = CapsAttentionLayer()(a4c_color_reshape)
    v_for_a4c_color = CapsuleLayer_for_final_model(
        name='capsule_output_for_a4c_color')(a4c_color_reshape)  # 1,2,16

    ###
    dense_for_RV_raw = tf.reshape(v_for_RV_raw, [-1, 32])
    dense_for_RV_color = tf.reshape(v_for_RV_color, [-1, 32])

    dense_for_SAX_raw = tf.reshape(v_for_SAX_raw, [-1, 32])
    dense_for_SAX_color = tf.reshape(v_for_SAX_color, [-1, 32])

    dense_for_a4c_raw = tf.reshape(v_for_a4c_raw, [-1, 32])
    dense_for_a4c_color = tf.reshape(v_for_a4c_color, [-1, 32])

    ensemble_vector = tf.keras.layers.Concatenate(
        axis=-1)([dense_for_RV_raw, dense_for_RV_color, dense_for_SAX_raw, dense_for_SAX_color, dense_for_a4c_raw, dense_for_a4c_color])  # None,64

    out_1 = Dense(1, activation='sigmoid', name="classification",
                  bias_initializer=tf.keras.initializers.Constant(output_bias))(ensemble_vector)

    # decoder part => 預計decode其中一種view ex. A4C color，目前打算在低階用 skip connection

    model = tf.keras.Model(inputs=[inp_RV, inp_RV_color, inp_SAX,
                           inp_SAX_color, inp_a4c, inp_a4c_color], outputs=[out_1])

    return model


async def load_weights(model, model_path):
    model.load_weights(model_path)
    return model


async def get_model_for_TR():
    event_detector_mod_task = asyncio.create_task(cnn_lstm())
    color_seg_task = asyncio.create_task(create_hourglass_network(
        3, 1, 128, (128, 128), (128, 128), bottleneck_mobile))  # inflow / outflow / bg

    RV_chamber_seg_task = asyncio.create_task(create_hourglass_network(
        3, 1, 128, (128, 128), (128, 128), bottleneck_mobile))  # atrium / ventricle / bg

    SAX_chamber_seg_task = asyncio.create_task(create_hourglass_network(
        4, 1, 128, (128, 128), (128, 128), bottleneck_mobile))  # atrium / ventricle / bg

    a4c_chamber_seg_task = asyncio.create_task(create_hourglass_network(
        5, 1, 128, (128, 128), (128, 128), bottleneck_mobile))  # atrium / ventricle / bg

    # 讀取模型架構
    event_detector_mod = await event_detector_mod_task
    color_seg = await color_seg_task
    RV_chamber_seg = await RV_chamber_seg_task
    SAX_chamber_seg = await SAX_chamber_seg_task
    a4c_chamber_seg = await a4c_chamber_seg_task

    event_detector_mod_task = asyncio.create_task(load_weights(event_detector_mod,
                                                               './ensemble_model/event_detector_best_model/event_detector_128.h5'))

    color_seg_task = asyncio.create_task(load_weights(color_seg,
                                                      './ensemble_model/color_regur_best_model/TR_4_128_model.h5'))
    RV_chamber_seg_task = asyncio.create_task(load_weights(RV_chamber_seg,
                                                           './ensemble_model/color_chamber_seg_best_model/RV_inflow_chamber_4_128_model.h5'))

    SAX_chamber_seg_task = asyncio.create_task(load_weights(SAX_chamber_seg,
                                                            './ensemble_model/color_chamber_seg_best_model/SAX_chamber_4_128_model.h5'))

    a4c_chamber_seg_task = asyncio.create_task(load_weights(a4c_chamber_seg,
                                                            './ensemble_model/color_chamber_seg_best_model/A4C_chamber_4_128_model.h5'))

    event_detector_mod = await event_detector_mod_task
    color_seg = await color_seg_task
    RV_chamber_seg = await RV_chamber_seg_task
    SAX_chamber_seg = await SAX_chamber_seg_task
    a4c_chamber_seg = await a4c_chamber_seg_task

    ensemble_mod = ensemble_model([-1.36660869], 32, False, event_detector_mod,
                                  color_seg, RV_chamber_seg,
                                  SAX_chamber_seg, a4c_chamber_seg)
    ensemble_mod.load_weights(
        './ensemble_model/all_ensemble_model_2/TR_sigmoid/best_model_TR_sigmoid')

    return ensemble_mod


'''

metrics={'classification': f1_metric}


ensemble_mod = ensemble_model(32 , False ,event_detector_mod,
                              color_seg, a2c_chamber_seg, a3c_chamber_seg, a4c_chamber_seg)

ensemble_mod.load_weights('best_model_MR')

'''
