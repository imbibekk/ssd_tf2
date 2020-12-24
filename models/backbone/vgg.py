import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.applications import VGG16

from .custom_layers import L2Normalization
from models import registry


WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def vgg16(pretrained=True): 
    
    inputs = layers.Input(shape=[None, None, 3])
    block1_conv1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(inputs)
    block1_conv2 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(block1_conv1)
    block1_pool = layers.MaxPool2D(2, 2, padding='same', name='block1_pool')(block1_conv2)

    block2_conv1 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(block1_pool)
    block2_conv2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(block2_conv1)
    block2_pool = layers.MaxPool2D(2, 2, padding='same', name='block2_pool')(block2_conv2)

    block3_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(block2_pool)
    block3_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(block3_conv1)
    block3_conv3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')(block3_conv2)
    block3_pool = layers.MaxPool2D(2, 2, padding='same', name='block3_pool')(block3_conv3)

    block4_conv1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(block3_pool)
    block4_conv2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(block4_conv1)
    block4_conv3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')(block4_conv2)
    block4_pool = layers.MaxPool2D(2, 2, padding='same', name='block4_pool')(block4_conv3)

    block5_conv1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')(block4_pool)
    block5_conv2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')(block5_conv1)
    block5_conv3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')(block5_conv2)

    vgg16 = tf.keras.models.Model(inputs = inputs, outputs=[block4_conv3, block5_conv3])
    
    if pretrained:
        print("Using pretrained weights of VGG16!!!")
        pretrained_weight = tf.keras.utils.get_file('vgg16', WEIGHTS_PATH_NO_TOP)
        vgg16.load_weights(pretrained_weight, by_name=True)

    return vgg16


def VGG16(vgg16):
    block4_conv3, block5_conv3 = vgg16.outputs
    block4_conv3 = L2Normalization(gamma_init=20)(block4_conv3)
    pool5 = layers.MaxPool2D(3, 1, padding='same')(block5_conv3)
    conv6 = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(pool5)
    conv7 = layers.Conv2D(1024, 1, padding='same', activation='relu')(conv6)
    
    model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=[block4_conv3, conv7])
    return model
    

def create_extra_layers(VGG16):
    
    out_38x38, out_19x19 = VGG16.outputs
    
    conv8_1 = layers.Conv2D(256, 1, activation='relu', padding='same')(out_19x19)
    conv8_pad = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(conv8_1)
    conv8_2 = layers.Conv2D(512, 3, strides=2, padding='valid', activation='relu')(conv8_pad)
    
    conv9_1 = layers.Conv2D(128, 1, activation='relu', padding='same')(conv8_2)
    conv9_pad = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(conv9_1)
    conv9_2 = layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu')(conv9_pad)

    conv10_1 = layers.Conv2D(128, 1, activation='relu', padding='same')(conv9_2)
    conv10_2 = layers.Conv2D(256, 3, activation='relu', padding='valid')(conv10_1)

    conv11_1 = layers.Conv2D(128, 1, activation='relu', padding='same')(conv10_2)
    conv11_2 = layers.Conv2D(256, 3, activation='relu', padding='valid')(conv11_1)

    model = tf.keras.models.Model(inputs= VGG16.inputs, outputs=[out_38x38, out_19x19, conv8_2, conv9_2, conv10_2, conv11_2])
    return model


class VGG(tf.keras.layers.Layer):
    def __init__(self, cfg=None, pretrained=True):
        super(VGG, self).__init__()
        vgg = vgg16(pretrained)
        fcs = VGG16(vgg)
        self.model = create_extra_layers(fcs)
    
    def call(self, x):
        outputs = self.model(x)
        return outputs


@registry.BACKBONES.register('vgg')
def vgg(cfg, pretrained=True):
    model = VGG(cfg, pretrained)
    return model


if __name__ == '__main__':
    import numpy as np
    vgg = VGG(pretrained=True)
    rand_inp = tf.random.normal([4, 300, 300, 3])
    outpus = vgg(rand_inp)

    for out in outpus:
        print(out.shape)