'''
Usage suggestion:

    import ae_model
    ae_model.register_custom_objects()
    ae_model.get_custom_objects()
    model = ae_model.get_model(config["model_parameter"])
'''
from custom_layers import *
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, TimeDistributed, GlobalAveragePooling2D, Cropping2D, LeakyReLU, ReLU, Lambda
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import deserialize, serialize

class Finalizer(tf.keras.layers.Layer):
    def __init__(self, paper_like_example, **kwargs):
        super().__init__(**kwargs)
        
        self.final_layers = []
        if paper_like_example:
            self.final_layers += [
                Conv2DRescaleBlock(conv2dblock={"num_filters":16, "pool":False, "spatial_dropout":True}, \
                                   rescale_height=58, rescale_width=58),
                Conv2DRescaleBlock(conv2dblock={"num_filters":1, "pool":False, "use_dropout":False}, \
                                     rescale_height=66, rescale_width=66)
            ]
        else:
            self.final_layers += [
                Conv2DTransposeBlock(num_filters=16, use_dropout=False),
                Conv2DTransposeBlock(num_filters=1, use_dropout=False),
                preprocessing.Resizing(70, 70, interpolation='bilinear'),
                Cropping2D(cropping=((3, 3), (3, 3))),
            ]
        self.final_layers += [
            #Lambda(lambda x: tf.keras.activations.sigmoid(x)),
            #LeakyReLU(),
        ]
        
    def call(self, x):
        for _layer in self.final_layers:
            x = _layer(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for _layer in self.final_layers:
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

def get_custom_objects():
    return tf.keras.utils.get_custom_objects()

def register_custom_objects():
    get_custom_objects()["Conv2DBlock"] = Conv2DBlock
    get_custom_objects()["Conv2DTransposeBlock"] = Conv2DTransposeBlock
    get_custom_objects()["Conv2DRescaleBlock"] = Conv2DRescaleBlock
    
    get_custom_objects()["Encoder"] = BuildingBlock
    get_custom_objects()["Decoder"] = BuildingBlock
    get_custom_objects()["Finalizer"] = Finalizer
    
    get_custom_objects()["EncoderInput"] = BuildingBlock
    get_custom_objects()["Output"] = BuildingBlock

def reset_custom_objects():
    get_custom_objects().clear()

def get_model(model_parameter, verbose=0):
    paper_like_example = True

    # build encoder
    encoder_input = deserialize(model_parameter["encoder_input"])
    input_ = tf.keras.Input(batch_shape=list(encoder_input._input_shape))
    encoder = TimeDistributed(deserialize(model_parameter["encoder"]), name="Encoder")
    encoder_output = encoder(encoder_input(input_))

    # build decoder
    decoder = deserialize(model_parameter["decoder"])
    output_ = tf.keras.Input(shape=list(decoder._input_shape))
    decoder = TimeDistributed(decoder, name="Decoder")
    finalizer = TimeDistributed(Finalizer(paper_like_example), name="Finalizer")
    output = deserialize(model_parameter["output"])
    decoder_output = output(finalizer(decoder(output_)))
    
    # create networks
    encoder_only = tf.keras.Model(inputs=input_, outputs=encoder_output, name="encoder_only")
    decoder_only = tf.keras.Model(inputs=output_, outputs=decoder_output, name="decoder_only")
    
    # compine both subnetworks
    autoencoder_output = decoder_only(encoder_only(input_))
    autoencoder = tf.keras.Model(inputs=input_, outputs=autoencoder_output, name="autoencoder")

    if verbose:
        print(autoencoder.summary())
        print(encoder_only.summary())
        print(decoder_only.summary())
        
    return autoencoder