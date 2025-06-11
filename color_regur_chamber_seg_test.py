# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 23:56:38 2022

@author: user
"""

import pickle
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input, TimeDistributed, Conv1D, Conv2D, Conv3D, MaxPool2D, UpSampling2D, Add, Dense, MaxPooling2D, Flatten, GRU, LSTM, Dropout, BatchNormalization
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# import cv2
import os
import matplotlib.pyplot as plt
import gzip
from tensorflow import keras
from tensorflow.keras import backend as K
import copy
import imageio
from collections import defaultdict


# %%
# event detector model
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


# Event 偵測
def cnn_lstm():  # 70%
    inp = Input(shape=(20, 256, 256, 3))
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
    x = Conv3D(256, (3, 3, 3), strides=(1, 2, 2),
               padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = ReshapeLayer([-1, 20, 1024])(x)

    x = LSTM(32, return_sequences=True, dropout=0.3)(x)  # return 32
    x = LSTM(32, return_sequences=True, dropout=0.3)(
        x)  # return 32 per time step (None, 20, 32)
    out = Conv1D(2, 3, padding='same', activation='sigmoid')(x)  # None, 20, 2
    mod = models.Model(inputs=inp, outputs=out)
    return mod


event_detector_mod = cnn_lstm()
event_detector_mod.load_weights(
    'seg_model/event_detector_best_model/event_detector.h5')


# %%
# seg model


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

    #x_out = tf.keras.layers.Activation('sigmoid', name = block_name + '_sigmoid_output')(x_out)

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

# Create three filter blocks per resolution.


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
    #head_parts = Conv2D(num_classes, kernel_size = (1, 1), activation = 'linear', padding = 'same', name = str(hgid) + 'conv_1x1_parts')(head)
    head_parts = Conv2D(num_classes, kernel_size=(
        1, 1), activation='softmax', padding='same', name=str(hgid) + 'conv_1x1_parts')(head)
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                  padding='same', name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear',
                    padding='same', name=str(hgid) + '_conv_1x1_x3')(head_parts)
    head_next_stage = Add()([head, head_m, prelayerfeatures])

    # 這裡試著加加看sigmoid layer
    #head_next_stage = tf.keras.layers.Conv2D(3,kernel_size=(1,1), activation = 'linear', padding = 'same', name = str(hgid) + '_output')(head_next_stage)

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


def create_front_module(input, num_channels, bottleneck):
    # front module, input to 1/4 of the resolution.
    # 1 (7, 7) conv2D + maxpool + 3 residual layers.
    # 加上 Cood_Conv layer
    x = add_coords_layer(input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same',
               activation='relu', name='front_conv_1x1_x1')(x)
    x = BatchNormalization(name='front')(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x1')
    #x = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)

    x = bottleneck(x, num_channels//2, 'front_residual_x2')
    x = bottleneck(x, num_channels, 'front_residual_x3')

    return x


def create_hourglass_network_for_seg(num_classes, num_stacks, num_channels, inres, outres, bottleneck):
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


# %%
class SegLayerRegur(tf.keras.layers.Layer):
    def __init__(self, model, out_shape, **kwargs):
        super(SegLayerRegur, self).__init__(**kwargs)

        self.model = model
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        # 不需要background，0 for regurg, 1 for inflow
        return self.model(inputs)[-1][:, :, :, 0:2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, self.out_shape)


class SegLayerChamber(tf.keras.layers.Layer):
    def __init__(self, model, out_shape, **kwargs):
        super(SegLayerChamber, self).__init__(**kwargs)

        self.model = model
        self.out_shape = out_shape

    def call(self, inputs, training=False):
        # 0 for atrium, 1 for ventricle
        return self.model(inputs)[-1][:, :, :, 0:2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, self.out_shape)


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
        diastole_output = tf.math.subtract(1.0, systole_output)  # 0,1,1,
        systole_diastole_concat = tf.concat(
            [systole_output, diastole_output], -1)

        chamber_output = tf.keras.layers.TimeDistributed(
            SegLayerChamber(self.chamber_seg, 2))(inputs)
        # channel 0 for left atrium, channel 1 for left ventricle
        chamber_output_ = tf.where(chamber_output > 0.5, 1.0, 0.0)

        color_output = tf.keras.layers.TimeDistributed(SegLayerRegur(self.seg_mod, 2))(
            inputs)  # channel 0 : regur / chnnel 1 : inflow

        out = tf.multiply(systole_diastole_concat,
                          color_output)  # None, 20, 256,256,2
        out_ = tf.multiply(chamber_output_, out)

        return out_

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, self.out_shape)  # 20,256,256,2
# %%


# %%
color_seg = create_hourglass_network_for_seg(
    3, 4, 128, (256, 256), (256, 256), bottleneck_mobile)  # inflow / outflow / bg
color_seg.load_weights(
    'seg_model/color_regur_best_model/MR_4_256_model.h5')

chamber_seg = create_hourglass_network_for_seg(
    3, 4, 128, (256, 256), (256, 256), bottleneck_mobile)  # atrium / ventricle / bg
chamber_seg.load_weights(
    'seg_model/color_chamber_seg_best_model/A3C_chamber_4_256_model.h5')

target = 'MV_inflow_TR'


def test_mod():
    inp = Input(shape=(20, 256, 256, 3))

    out = SystoleSeg(color_seg, chamber_seg, event_detector_mod, 2)(inp)
    mod = models.Model(inputs=inp, outputs=out)
    return mod


test = test_mod()


# %% test color data

with open(r'E:\\classified_results\\file_and_class.pickle', 'rb') as f:
    RawData_TR_npz = pickle.load(f)
# type(RawData_TR_npz)
# RawData_TR_npz['160017CH0002_1_20.dcm.npz']
class2names = defaultdict(list)
for k, v in RawData_TR_npz.items():
    class2names[v].append(k)


TARGET_POSE2color_cv2 = {
    0: (255, 0, 0),  # '#ff0000',
    1: (0, 0, 255),  # '#0000ff',
    2: (0, 255, 0),  # '#00ff00',
    3: (255, 255, 0),  # '#ffff00',
    4: (196, 0, 255)  # '#C400FF'

}

TARGET_POSE2color_plt = {
    0: '#ff0000',
    1: '#0000ff',
    2: '#00ff00',
    3: '#ffff00',
    4: '#C400FF'

}

TARGET_POSE2color_plt = {
    0: '#ff0000',
    1: '#0000ff',
    2: '#00ff00',
    3: '#ffff00',
    4: '#C400FF'

}


h, w, c = 300, 400, 3
start_x = w//2 - 10 - 128
start_y = h//2 + 13 - 128

# Channel分開看
is_split_vide_mode = True
drawing_contour = True

data = []

# For color segmentation
# data.extend(class2names['Apical 4-chamber -color'][0:20])
# data.extend(class2names['Apical 2-chamber-color'][0:20])
# data.extend(class2names['Apical 3-chamber-color'][0:20])


# severe mr
target = 'a2c'
severe_tr_id = ["166018K00253"]
#moderate_mr_id = ["160018P51271"]
data = []
for i in class2names['Apical 2-chamber–color']:
    if i[:-13] in severe_tr_id:
        data.append(i)


data


# %%
# without raw image
for i, _data in enumerate(data):

    imgs = np.load(r'E:\RawData_TR_npz\\' + _data)['arr_0']
    resize_imgs_256 = []
    for j in range(imgs.shape[0]):
        img = imgs[j]
        resize_img = cv2.resize(
            img, (400, 300), interpolation=cv2.INTER_CUBIC)/255
        resize_img_256 = resize_img[start_y:start_y +
                                    256, start_x:start_x+256, :]
        resize_imgs_256.append(resize_img_256)

    resize_imgs_256_ = np.expand_dims(np.array(resize_imgs_256)[
                                      :20], 0)  # 符合 None, 20, 256, 256, 3

    predicted_result = np.squeeze(
        test.predict(resize_imgs_256_))  # None, 20, 256, 256, 2

    image_gif_filename = []

    print(f'imgs.shape => {imgs.shape}')
    for frame in range(20):

        copied_resize_imgs_256 = copy.deepcopy(resize_imgs_256[frame])
        copied_resize_imgs_256 = (
            copied_resize_imgs_256 * 255).astype(np.uint8)

        colored_target_predicted_result_ = np.zeros((256, 256, 3))
        for seg_idx in range(2):  # 0 for regur 1 for inflow
            target_predicted_result = predicted_result[frame][:, :, seg_idx]
            target_predicted_result = target_predicted_result.reshape(
                256, 256, 1)

            r = copy.deepcopy(target_predicted_result)
            g = copy.deepcopy(target_predicted_result)
            b = copy.deepcopy(target_predicted_result)

            TARGET_POSE2color_cv2_index = seg_idx  # 針對不同的seg_idx配上不同顏色

            # (0~1 in 256*256)*(255)
            r = (
                r*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][0]).astype(np.uint8)
            g = (
                g*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][1]).astype(np.uint8)
            b = (
                b*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][2]).astype(np.uint8)

            colored_target_predicted_result = np.concatenate((r, g, b), 2)

            #copied_resize_imgs_256 = (copied_resize_imgs_256 * 1.) + (colored_target_predicted_result * 0.8)
            #copied_resize_imgs_256 = colored_target_predicted_result * 1
            colored_target_predicted_result_ += colored_target_predicted_result
            # 這裡已經都轉為255
        colored_target_predicted_result = colored_target_predicted_result_.astype(
            np.uint8)
        gauss_result = cv2.GaussianBlur(
            colored_target_predicted_result, (5, 5), 0)
        #copied_resize_imgs_256 = copied_resize_imgs_256.astype('uint8')
        plt.title(f'{_data}_{frame}')
        plt.imshow(gauss_result)
        # plt.imshow(copied_resize_imgs_256)

        file_path = rf"C:\Users\user\Desktop\temp\{_data}_{frame}.png"
        plt.savefig(file_path, bbox_inches='tight', dpi=200)
        plt.close()
        image_gif_filename.append(file_path)

    gif = []
    for f in image_gif_filename:
        gif.append(imageio.imread(f))

    imageio.mimsave(
        rf'C:\\Users\\user\\Desktop\\gif\\{target}_{_data}_severe_mr_new.gif', gif, duration=0.2)

    for f in image_gif_filename:
        os.remove(f)

# %%
# with raw image
for i, _data in enumerate(data):

    imgs = np.load(r'E:\RawData_TR_npz\\' + _data)['arr_0']
    resize_imgs_256 = []
    for j in range(imgs.shape[0]):
        img = imgs[j]
        resize_img = cv2.resize(
            img, (400, 300), interpolation=cv2.INTER_CUBIC)/255
        resize_img_256 = resize_img[start_y:start_y +
                                    256, start_x:start_x+256, :]
        resize_imgs_256.append(resize_img_256)

    resize_imgs_256_ = np.expand_dims(np.array(resize_imgs_256)[
                                      :20], 0)  # 符合 None, 20, 256, 256, 3

    predicted_result = np.squeeze(test.predict(
        resize_imgs_256_))  # None, 20, 256, 256, 2

    image_gif_filename = []

    print(f'imgs.shape => {imgs.shape}')
    for frame in range(20):

        copied_resize_imgs_256 = copy.deepcopy(resize_imgs_256[frame])
        copied_resize_imgs_256 = (
            copied_resize_imgs_256 * 255).astype(np.uint8)

        colored_target_predicted_result_ = np.zeros((256, 256, 3))
        for seg_idx in range(2):  # 0 for regur 1 for inflow
            target_predicted_result = predicted_result[frame][:, :, seg_idx]
            target_predicted_result = target_predicted_result.reshape(
                256, 256, 1)

            r = copy.deepcopy(target_predicted_result)
            g = copy.deepcopy(target_predicted_result)
            b = copy.deepcopy(target_predicted_result)

            TARGET_POSE2color_cv2_index = seg_idx  # 針對不同的seg_idx配上不同顏色

            # (0~1 in 256*256)*(255)
            r = (
                r*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][0]).astype(np.uint8)
            g = (
                g*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][1]).astype(np.uint8)
            b = (
                b*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][2]).astype(np.uint8)

            colored_target_predicted_result = np.concatenate((r, g, b), 2)

            #copied_resize_imgs_256 = (copied_resize_imgs_256 * 1.) + (colored_target_predicted_result * 0.8)
            #copied_resize_imgs_256 = colored_target_predicted_result * 1
            colored_target_predicted_result_ += colored_target_predicted_result
            # 這裡已經都轉為255
        colored_target_predicted_result = colored_target_predicted_result_.astype(
            np.uint8)

        gauss_result = cv2.GaussianBlur(
            colored_target_predicted_result, (5, 5), 0)

        #final_result = cv2.addWeighted(copied_resize_imgs_256,0.5,gauss_result,0.5,0)
        final_result = copied_resize_imgs_256 * 1. + gauss_result * 0.8
        final_result = (final_result - final_result.min()) / \
            (final_result.max() - final_result.min())
        #copied_resize_imgs_256 = copied_resize_imgs_256.astype('uint8')
        plt.title(f'{_data}_{frame}')
        plt.imshow(final_result)
        # plt.imshow(copied_resize_imgs_256)

        file_path = rf"C:\Users\user\Desktop\temp\{_data}_{frame}.png"
        plt.savefig(file_path, bbox_inches='tight', dpi=200)
        plt.close()
        image_gif_filename.append(file_path)

    gif = []
    for f in image_gif_filename:
        gif.append(imageio.imread(f))

    imageio.mimsave(
        rf'C:\\Users\\user\\Desktop\\gif\\{target}_{_data}_severe_mr_with_test.gif', gif, duration=0.2)

    for f in image_gif_filename:
        os.remove(f)

# %%
# new method
for i, _data in enumerate(data):

    imgs = np.load(r'E:\RawData_TR_npz\\' + _data)['arr_0']
    resize_imgs_256 = []
    for j in range(imgs.shape[0]):
        img = imgs[j]
        resize_img = cv2.resize(
            img, (400, 300), interpolation=cv2.INTER_CUBIC)/255
        resize_img_256 = resize_img[start_y:start_y +
                                    256, start_x:start_x+256, :]
        resize_imgs_256.append(resize_img_256)

    resize_imgs_256_ = np.expand_dims(np.array(resize_imgs_256)[
                                      :20], 0)  # 符合 None, 20, 256, 256, 3

    predicted_result = np.squeeze(test.predict(
        resize_imgs_256_))  # None, 20, 256, 256, 2

    image_gif_filename = []

    print(f'imgs.shape => {imgs.shape}')
    for frame in range(20):

        copied_resize_imgs_256 = copy.deepcopy(resize_imgs_256[frame])
        copied_resize_imgs_256 = (
            copied_resize_imgs_256 * 255).astype(np.uint8)

        copied_resize_imgs_256_seg = copy.deepcopy(copied_resize_imgs_256)

        colored_target_predicted_result_ = np.zeros((256, 256, 3))
        colored_seg_ = np.zeros((256, 256, 3))
        for seg_idx in range(2):  # 0 for regur 1 for inflow
            target_predicted_result_0 = predicted_result[frame][:, :, seg_idx]
            target_predicted_result = target_predicted_result_0.reshape(
                256, 256, 1)

            #

            r = copy.deepcopy(target_predicted_result)
            g = copy.deepcopy(target_predicted_result)
            b = copy.deepcopy(target_predicted_result)

            r_ = (r*255).astype(np.uint8)
            g_ = (g*255).astype(np.uint8)
            b_ = (b*255).astype(np.uint8)

            TARGET_POSE2color_cv2_index = seg_idx  # 針對不同的seg_idx配上不同顏色

            # (0~1 in 256*256)*(255)
            r = (
                r*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][0]).astype(np.uint8)
            g = (
                g*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][1]).astype(np.uint8)
            b = (
                b*TARGET_POSE2color_cv2[TARGET_POSE2color_cv2_index][2]).astype(np.uint8)

            colored_target_predicted_result = np.concatenate((r, g, b), 2)

            colored_seg = np.concatenate((r_, g_, b_), 2)

            colored_seg_ += colored_seg
            colored_target_predicted_result_ += colored_target_predicted_result
            # 這裡已經都轉為255
        colored_target_predicted_result = colored_target_predicted_result_.astype(
            np.uint8)

        gauss_result = cv2.GaussianBlur(
            colored_target_predicted_result, (5, 5), 0)

        mask = np.where(colored_seg_ > 0)
        mask_0 = np.where(colored_seg_ == 0)

        copied_resize_imgs_256[mask] = 0
        copied_resize_imgs_256_seg[mask_0] = 0  # add weight with seg_result

        #mask = np.where(gauss_result > 0)
        #mask_0 = np.where(gauss_result == 0)

        #copied_resize_imgs_256[mask] = 0
        # copied_resize_imgs_256_seg[mask_0] = 0 # add weight with seg_result

        seg_output = cv2.addWeighted(
            copied_resize_imgs_256_seg, 0.3, gauss_result, 0.7, 0)

        #final_result = cv2.addWeighted(copied_resize_imgs_256,0.5,gauss_result,0.5,0)
        final_result = copied_resize_imgs_256 + seg_output

        #copied_resize_imgs_256 = copied_resize_imgs_256.astype('uint8')
        plt.title(f'{_data}_{frame}')
        plt.imshow(final_result)
        # plt.imshow(copied_resize_imgs_256_seg)

        file_path = rf"C:\Users\user\Desktop\temp\{_data}_{frame}.png"
        plt.savefig(file_path, bbox_inches='tight', dpi=200)
        plt.close()
        image_gif_filename.append(file_path)

    gif = []
    for f in image_gif_filename:
        gif.append(imageio.imread(f))

    imageio.mimsave(
        rf'C:\\Users\\user\\Desktop\\gif\\{target}_{_data}_severe_mr_with_test_1.gif', gif, duration=0.2)

    for f in image_gif_filename:
        os.remove(f)


# %%
# experiment
