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
import ae_model

class Network():
    def __init__(self, *args, verbose=0, **kwargs):
        self._model_parameter, self._training_parameter = args
        self._model_path = "./models/%s/" % str(datetime.datetime.now()).replace(" ","_")
        self._verbose = verbose
        
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
                          loss=self.loss(None, None),
                          loss_weights={"decoder":1, "classifier": 0.005},
                          metrics=self.metrics(None, None))
        
    def save(self):
        #create model directory first
        os.makedirs(self._model_path, exist_ok=True)
        try:
            self._model.save("%sweights.%05d" % (self._model_path, self._epoch), save_format="tf")
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
        self._model = tf.keras.models.load_model("%sweights.%05d" % (model_path, epoch))
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
        ae_model.register_custom_objects()
        #ae_model.get_custom_objects()
        number_classes = 10
        
        # initial definition of the sequential model
        (self._encoder_input, self._encoder_only) = ae_model.create_encoder(self._model_parameter["encoder_input"], self._model_parameter["encoder"])
        (self._decoder_input, self._decoder_only) = ae_model.create_decoder(self._model_parameter["decoder"], self._model_parameter["finalizer"], self._model_parameter["output"])
        
        # classifier
        classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(number_classes, activation='softmax')
        ], name="classifier")
        
        encoder = self._encoder_only(self._encoder_input)
        # combine networks
        autoencoder_output = self._decoder_only(encoder)
        # classifier output
        classifier_output = classifier(encoder)
        
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
        
    def loss(self, y_true, y_pred):
        # calc mean over the batch dimension
        #return K.mean(self._calc_loss(y_true, y_pred), axis=0, keepdims=True)
        return { "decoder": "mae", "classifier": "categorical_crossentropy", }
        
    def metrics(self, y_true, y_pred):
        # return <some metrics>
        metrics = {}
        metrics["decoder"] = ["mse"]
        metrics["decoder"] += ["mae"]
        metrics["classifier"] = ["accuracy"]
        return metrics
    
    def _calc_loss(self, labels, predictions):
        #return <loss for each and every example>
        bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE,
                    name='binary_crossentropy')
        return bce(labels, predictions, sample_weight=None)