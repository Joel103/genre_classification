'''
Usage suggestion:

    from ae_model import get_model
    model = get_model(verbose=1)
'''

import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, TimeDistributed, Lambda, MaxPool2D, UpSampling2D, \
                                        GlobalAveragePooling2D, Cropping2D, LeakyReLU, ReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D, Dropout
from tensorflow.keras.layers.experimental import preprocessing

# model params
dropout_rate = 0.1
spatial_dropout_rate = 0.1
linear = Lambda(lambda x: x)

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, conv_stride=1, pool_size=2, pool_stride=2, \
                 padding="VALID", pool=True, activate=True, use_dropout=True, spatial_dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        
        linear = Lambda(lambda x: x)
        self.conv = TimeDistributed(Conv2D(num_filters, kernel_size, (conv_stride, conv_stride), padding))
        self.pool = TimeDistributed(MaxPool2D((pool_size,pool_size), (pool_stride,pool_stride))) if pool \
            else linear
        self.acti = TimeDistributed(LeakyReLU()) if activate else linear
        if use_dropout:
            self.regu = TimeDistributed(SpatialDropout2D(spatial_dropout_rate)) if spatial_dropout \
                else TimeDistributed(Dropout(dropout_rate))
        else:
            self.regu = linear
            
    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.acti(x)
        x = self.regu(x)
        return x
        
class Conv2DTransposeBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, conv_stride=1, pool_size=2, activate=True, \
                 use_dropout=True, spatial_dropout=False, interpolation="nearest", **kwargs):
        super().__init__(**kwargs)
        
        self.pool = TimeDistributed(UpSampling2D((pool_size, pool_size), interpolation=interpolation))
        self.conv = TimeDistributed(Conv2DTranspose(num_filters, kernel_size, strides=(conv_stride, conv_stride)))
        self.acti = TimeDistributed(LeakyReLU()) if activate else linear
        if use_dropout:
            self.regu = TimeDistributed(SpatialDropout2D(spatial_dropout_rate)) if spatial_dropout \
                else TimeDistributed(Dropout(dropout_rate))
        else:
            self.regu = linear
            
    def call(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.acti(x)
        x = self.regu(x)
        return x
    
class RescaleConvolveBlock(tf.keras.layers.Layer): 
    def __init__(self, num_filters=16, kernel_size=3, conv_stride=1, pool_size=2, pool_stride=2, \
                 padding="VALID", pool=True, activate=True, use_dropout=True, spatial_dropout=False, \
                 rescale_height=64, rescale_width=64, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        
        self.pool = TimeDistributed(preprocessing.Resizing(rescale_height, rescale_width, \
                                                           interpolation=interpolation))
        self.conv = Conv2DBlock(num_filters, kernel_size, conv_stride, pool_size, pool_stride, \
                                padding=padding, pool=pool, use_dropout=use_dropout)
        
    def call(self, x):
        x = self.pool(x)
        return self.conv(x)
        
class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super().__init__(**kwargs)
        
        # encoder
        self.reshape = Reshape((-1, 64, 64, 1))            
        self.convblock0 = Conv2DBlock(32, 3, 1, 2, 2, spatial_dropout=True)
        self.convblock1 = Conv2DBlock(64, 3, 1, 2, 2, spatial_dropout=True)
        self.convblock2 = Conv2DBlock(128, 3, 1, 2, 2)
        self.convblock3 = Conv2DBlock(embedding_size, 3, 1, 2, 2)

        # embedding
        self.embedding = TimeDistributed(GlobalAveragePooling2D())

    def call(self, x):
        x = self.reshape(x)
        x = self.convblock0(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return self.embedding(x)
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super().__init__(**kwargs)

        # decoder
        self.reshape = Reshape((-1, 1, 1, embedding_size))
        self.convblock0 = Conv2DTransposeBlock(128, 3, 1, 2)
        self.convblock1 = Conv2DTransposeBlock(64, 3, 1, 2)
        self.convblock2 = Conv2DTransposeBlock(32, 3, 1, 2)
        self.convblock3 = Conv2DTransposeBlock(32, 3, 1, 2)

    def call(self, x):
        x = self.reshape(x)
        x = self.convblock0(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class Finalizer(tf.keras.layers.Layer):
    def __init__(self, paper_like_example, input_shape, **kwargs):
        super().__init__(**kwargs)
        
        self.final_layers = []
        if paper_like_example:
            self.final_layers += [
                RescaleConvolveBlock(16, 3, 1, 2, 2, padding="VALID", pool=False, use_dropout=True, \
                                     spatial_dropout=True, rescale_height=58, rescale_width=58),
                RescaleConvolveBlock(1, 3, 1, 2, 2, padding="VALID", pool=False, use_dropout=False, \
                                     rescale_height=66, rescale_width=66)
            ]
        else:
            self.final_layers += [
                Conv2DTransposeBlock(16, 3, 1, 2, use_dropout=False),
                Conv2DTransposeBlock(1, 3, 1, 2, use_dropout=False),
                TimeDistributed(preprocessing.Resizing(70, 70, interpolation='bilinear')),
                TimeDistributed(Cropping2D(cropping=((3, 3), (3, 3)))),
            ]
        self.final_layers += [
            Reshape(input_shape),
            #Lambda(lambda x: tf.keras.activations.sigmoid(x)),
            ReLU(),
        ]
        
    def call(self, x):
        for layer in self.final_layers:
            x = layer(x)
        return x

def get_model(verbose=0):
    input_shape = (512, 64)
    embedding_size=256

    # https://distill.pub/2016/deconv-checkerboard/
    paper_like_example = True

    layers = [
        Input(shape=input_shape),
        Encoder(embedding_size),
        Decoder(embedding_size),
        Finalizer(paper_like_example, input_shape),
    ]

    model = tensorflow.keras.models.Sequential(layers)
    if verbose:
        print(model.summary())

    return model
