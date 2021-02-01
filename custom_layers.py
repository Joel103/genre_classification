import tensorflow as tf
from tensorflow.keras.layers import Reshape, Lambda, MaxPool2D, UpSampling2D, LeakyReLU, ReLU, Cropping2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import deserialize

# model params
linear = Lambda(lambda x: x)

class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, conv_stride=1, pool_size=2, pool_stride=2, \
                 padding="VALID", pool=True, activate=True, use_dropout=True, spatial_dropout=False, \
                 dropout_rate=0.1, spatial_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.padding = padding
        self.pool = pool
        self.activate = activate
        self.use_dropout = use_dropout
        self.spatial_dropout = spatial_dropout
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
    
        self.conv = Conv2D(self.num_filters, self.kernel_size, (self.conv_stride, self.conv_stride), self.padding)
        self.pool = MaxPool2D((self.pool_size, self.pool_size), (self.pool_stride, self.pool_stride)) \
                        if self.pool else linear
        self.acti = LeakyReLU() if self.activate else linear
        
        if self.use_dropout:
            self.regu = SpatialDropout2D(self.spatial_dropout_rate) if self.spatial_dropout \
                else Dropout(self.dropout_rate)
        else:
            self.regu = linear
            
    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.acti(x)
        x = self.regu(x)
        return x
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.conv.compute_output_shape(output_shape)
        output_shape = self.pool.compute_output_shape(output_shape)
        output_shape = self.acti.compute_output_shape(output_shape)
        output_shape = self.regu.compute_output_shape(output_shape)
        return output_shape
    
    def get_config(self):
        return {"num_filter": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_stride": self.conv_stride,
                "pool_size": self.pool_size,
                "pool_stride": self.pool_stride,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
               }

class Conv2DTransposeBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, conv_stride=1, pool_size=2, activate=True, \
                 use_dropout=True, spatial_dropout=False, interpolation="nearest", \
                 dropout_rate=0.1, spatial_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size = pool_size
        self.activate = activate
        self.use_dropout = use_dropout
        self.spatial_dropout = spatial_dropout
        self.interpolation = interpolation
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        
        self.pool = UpSampling2D((self.pool_size, self.pool_size), interpolation=self.interpolation)
        self.conv = Conv2DTranspose(self.num_filters, self.kernel_size, strides=(self.conv_stride, self.conv_stride))
        self.acti = LeakyReLU() if self.activate else linear
        if self.use_dropout:
            self.regu = SpatialDropout2D(self.spatial_dropout_rate) if self.spatial_dropout \
                else Dropout(self.dropout_rate)
        else:
            self.regu = linear
            
    def call(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.acti(x)
        x = self.regu(x)
        return x
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.pool.compute_output_shape(output_shape)
        output_shape = self.conv.compute_output_shape(output_shape)
        output_shape = self.acti.compute_output_shape(output_shape)
        output_shape = self.regu.compute_output_shape(output_shape)
        return output_shape
    
    def get_config(self):
        return {"num_filter": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_stride": self.conv_stride,
                "pool_size": self.pool_size,
                "pool_stride": self.pool_stride,
                "dropout_rate": self.dropout_rate,
                "spatial_dropout_rate": self.spatial_dropout_rate,
               }
    
# https://distill.pub/2016/deconv-checkerboard/
class Conv2DRescaleBlock(tf.keras.layers.Layer): 
    def __init__(self, conv2dblock, rescale_height=64, rescale_width=64, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        
        self.rescale_height = rescale_height
        self.rescale_width = rescale_width
        self.interpolation = interpolation
        self.conv2dblock = conv2dblock
        
        self.pool = preprocessing.Resizing(rescale_height, rescale_width, interpolation=interpolation)
        self.conv = Conv2DBlock(**conv2dblock)
        
    def call(self, x):
        x = self.pool(x)
        return self.conv(x)
        
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.pool.compute_output_shape(output_shape)
        output_shape = self.conv.compute_output_shape(output_shape)
        return output_shape
    
    def get_config(self):
        return {"rescale_height": self.rescale_height,
                "rescale_width": self.rescale_width,
                "interpolation": self.interpolation,
                "conv2dblock": self.conv2dblock,
               }

class BuildingBlock(tf.keras.layers.Layer):
    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers_config = layers
              
        sorted_keys = sorted(list([*self.layers_config]))
        self.layers = []
        for key in sorted_keys:
            self.layers += [deserialize(self.layers_config[key])]
        
        self._input_shape = list(self.layers[0].input_shape[0])

    def call(self, x):
        for _layer in self.layers:
            x = _layer(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for _layer in self.layers:
            # Workaround: Tensorflow Bug(?) "tf.keras.layers.InputLayer" has no method "compute_output_shape"
            if isinstance(_layer, tf.keras.layers.InputLayer):
                output_shape = _layer._batch_input_shape
            else:
                output_shape = _layer.compute_output_shape(output_shape)
        return output_shape
    
    def get_config(self):
        return {"layers": self.layers_config}
    

class Finalizer(tf.keras.layers.Layer):
    def __init__(self, paper_like_example, **kwargs):
        super().__init__(**kwargs)
        
        self.layers = []
        if paper_like_example:
            self.layers += [
                Conv2DRescaleBlock(conv2dblock={"num_filters":16, "pool":False, "spatial_dropout":True}, \
                                   rescale_height=58, rescale_width=58),
                Conv2DRescaleBlock(conv2dblock={"num_filters":1, "pool":False, "use_dropout":False}, \
                                     rescale_height=66, rescale_width=66)
            ]
        else:
            self.layers += [
                Conv2DTransposeBlock(num_filters=16, use_dropout=False),
                Conv2DTransposeBlock(num_filters=1, use_dropout=False),
                preprocessing.Resizing(70, 70, interpolation='bilinear'),
                Cropping2D(cropping=((3, 3), (3, 3))),
            ]
        self.layers += [
            #Lambda(lambda x: tf.keras.activations.sigmoid(x)),
            #LeakyReLU(),
        ]
        
    def call(self, x):
        for _layer in self.layers:
            x = _layer(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for _layer in self.layers:
            if isinstance(_layer, tf.keras.layers.InputLayer):
                output_shape = _layer._batch_input_shape
            else:
                output_shape = _layer.compute_output_shape(output_shape)
        return output_shape

    # TODO: the following 2 methods
    def get_config(self):
        return {}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)