"""Neural Network class with architecture and loss functions.

  Typical usage example:

  network = Network(model_parameter, training_parameter)
  network.compile()
  network.save()

  or
  
  network = Network(model_parameter, training_parameter)
  network.load_model(10000, "models/")
  network.compile()

"""
from tensorflow import keras as keras
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
import datetime
import os
from resnet import *
from resnet_decoder import *

class Network():
    def __init__(self, *args, verbose=0, **kwargs):
        self._model_parameter, self._training_parameter = args
        self._model_path = "./models/%s/" % str(datetime.datetime.now()).replace(" ","_")
        self._verbose = verbose
        
        self._embedding_size = self._model_parameter["embedding_size"]
        self._input_shape = self._model_parameter["input_shape"]
        self._num_classes = self._model_parameter["num_classes"]
        
        self._loss = self._training_parameter["loss"]
        self._metrics = self._training_parameter["metrics"]
        self._loss_weights = self._training_parameter["loss_weights"]
        
        # setting the precision to mixed if float16 is desired in training_parameter
        self._set_precision(self._training_parameter["calculation_dtype"], self._training_parameter["calculation_epsilon"])

        # assemble network
        self._create_model()
        
        if self._verbose:
            # print the model properties
            self._model.summary()

        # set an epoch
        self._epoch = 0
        
        self._set_optimizer()
        
        # creating list with some important callbacks
        self._callbacks = []
        self._add_callbacks()

    def compile(self):
        # compile model
        self._model.compile(optimizer=self._optimizer,
                          loss=self._loss,
                          loss_weights=self._loss_weights,
                          metrics=self._metrics)
        
    def save(self):
        #create model directory first
        os.makedirs(self._model_path, exist_ok=True)
        os.makedirs(f"{self._model_path}weights", exist_ok=True)
        try:
            self._model.save("%sweights/%05d" % (self._model_path, self._epoch), save_format="tf")
            if self._verbose:
                print("Saved model to disk")
        except RuntimeError:
            if self._verbose:
                print("Couldn't save model to disk")
    
    def load_model(self, epoch=-1, model_path=None):
        if epoch < 0:
            epoch = self._epoch
        if model_path is None:
            model_path = self._model_path
        
        # load weights into new model
        self._model = tf.keras.models.load_model("%sweights/%05d" % (model_path, epoch))
        self._encoder_only = [layer for layer in self._model.layers if "encoder_only" in layer._name][0]
        self._epoch = epoch
        self._model_path = model_path
        
        if self._verbose:
            print("Loaded model from disk")
    
    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)
    
    def predict_on_batch(self, *args, **kwargs):
        return self._model.predict_on_batch(*args, **kwargs)
    
    def predict_embedding_on_batch(self, *args, **kwargs):
        return self._encoder_only.predict_on_batch(*args, **kwargs)
    
    @property
    def callbacks(self):
        return self._callbacks
    
    @property
    def epoch(self):
        return self._epoch
    
    @epoch.setter
    def epoch(self, value):
        self._epoch = value
    
    @property
    def model_path(self):
        return self._model_path
    
    @property
    def config_path(self):
        return self._model_path + "config.json"
        
    @staticmethod
    def _set_precision(calculation_dtype,calculation_epsilon):
        # enable single/half/double precision
        import tensorflow.keras.backend as K
        K.set_floatx(calculation_dtype)
        K.set_epsilon(calculation_epsilon)

        # enable mixed precission
        if "float16" in calculation_dtype:
            import tensorflow.keras.mixed_precision as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            
    def _create_model(self):
        
        model_input_shape = self._input_shape
        channel_dims = self._embedding_size

        # encoder network
        self._encoder_input = tf.keras.Input(shape=model_input_shape)
        resnet = ResNet18(num_classes=channel_dims)
        encoder_output = tf.keras.layers.Reshape((1, channel_dims))(resnet(self._encoder_input))
        self._encoder_only = tf.keras.Model(inputs=self._encoder_input, outputs=encoder_output, name="encoder_only")

        # decoder network
        self._decoder_input = tf.keras.Input(shape=[None, channel_dims])
        decoder = Basic_Decoder()
        decoder_output = tf.keras.layers.Reshape(model_input_shape)(decoder(tf.keras.layers.Reshape((1, 1, channel_dims))(self._decoder_input)))
        self._decoder_only = tf.keras.Model(inputs=self._decoder_input, outputs=decoder_output, name="decoder_only")

        # classifier
        classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self._num_classes, activation='softmax')
        ], name="classifier_only")
        
        # combine networks
        autoencoder_output = self._decoder_only(encoder_output)
        # classifier output
        classifier_output = classifier(encoder_output)
        
        autoencoder = tf.keras.Model(inputs=self._encoder_input, 
                                     outputs={"decoder": autoencoder_output, "classifier": classifier_output}, name="autoencoder")
        self._model = autoencoder
        
    def _add_callbacks(self):
        from callbacks import IncreaseEpochCustom, SaveEveryNthEpochCustom
        self._callbacks += [IncreaseEpochCustom(self)]
        self._callbacks += [SaveEveryNthEpochCustom(self, self._training_parameter["save_steps"])]
        # tensorboard callback
        self._callbacks += [tf.keras.callbacks.TensorBoard(log_dir=self._model_path+"/logs",
                                                     histogram_freq=0, write_graph=True,
                                                     write_images=False, update_freq="epoch",
                                                     profile_batch=0, embeddings_freq=0, 
                                                     embeddings_metadata=None)]
    def _set_optimizer(self):
        # Catching the: ValueError: Gradient clipping in the optimizer (by setting clipnorm or clipvalue) is currently unsupported when using a distribution strategy.
        is_distributed = tf.distribute.in_cross_replica_context()

        clip_gradient = "optimizer_clip_parameter" in self._training_parameter
        is_half_precision = "float16" in self._training_parameter["calculation_dtype"]

        optimizer_clip_parameter = {}
        if clip_gradient and self._training_parameter["optimizer_clip_parameter"] is not None \
                            and not is_half_precision and not is_distributed: 
            optimizer_clip_parameter = self._training_parameter["optimizer_clip_parameter"]
            
        optimizer_config = { "class_name" : self._training_parameter["optimizer_class_name"], 
                             "config" : { "learning_rate" : self._training_parameter["learning_rate"],
                                          **self._training_parameter["optimizer_parameter"],
                                          **optimizer_clip_parameter
                                        }
                            }
        
        self._optimizer = keras.optimizers.get(optimizer_config)