from custom_layers import *
import tensorflow as tf
from tensorflow.keras.layers import deserialize

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
    get_custom_objects()["BuildingBlock"] = BuildingBlock

def reset_custom_objects():
    get_custom_objects().clear()
    
def create_encoder(encoder_input_config, encoder_config):
    # build encoder
    encoder_input = deserialize(encoder_input_config)
    _encoder_input = tf.keras.Input(batch_shape=list(encoder_input._input_shape))
    encoder = tf.keras.layers.TimeDistributed(deserialize(encoder_config), name="Encoder")
    encoder_output = encoder(encoder_input(_encoder_input))
    # create network
    _encoder_only = tf.keras.Model(inputs=_encoder_input, outputs=encoder_output, name="encoder_only")
    
    return (_encoder_input, _encoder_only)

def create_decoder(decoder_config, finalizer_config, output_config):
    paper_like_example = False

    # build decoder
    decoder = deserialize(decoder_config)
    _decoder_input = tf.keras.Input(shape=list(decoder._input_shape))
    decoder = tf.keras.layers.TimeDistributed(decoder, name="Decoder")
    # TODO: finalizer_config
    finalizer = tf.keras.layers.TimeDistributed(Finalizer(paper_like_example), name="Finalizer")
    output = deserialize(output_config)
    decoder_output = output(finalizer(decoder(_decoder_input)))
    # create network
    _decoder_only = tf.keras.Model(inputs=_decoder_input, outputs=decoder_output, name="decoder_only")
    
    return (_decoder_input, _decoder_only)