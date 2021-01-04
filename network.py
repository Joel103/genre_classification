# Copyright (c) 2020, Nico Jahn
# All rights reserved.

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
# TODO: import all other layers here
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
import datetime
import os

class Network():
    def __init__(self, *args, verbose=0, **kwargs):
        self._model_parameter, self._training_parameter = args
        self._model_path = "./models/%s/" % str(datetime.datetime.now()).replace(" ","_")
        self._verbose = verbose
        
        # setting the precision to mixed if float16 is desired in training_parameter
        self._set_precision(self._training_parameter["calculation_dtype"], self._training_parameter["calculation_epsilon"])
        
        # assemble network
        self._create_model()
        self._add_input()
        self._add_core()
        self._add_output()
        
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
                          loss=self.loss,
                          metrics=self.metrics)
        
    def save(self):
        #create model directory first
        os.makedirs(self._model_path, exist_ok=True)
        # serialize model to JSON
        model_json = self._model.to_json()
        with open("%smodel.json" % (self._model_path), "w") as json_file:
            json_file.write(model_json)
        try:
            # serialize weights to HDF5
            self._model.save_weights("%sweights.%05d.hdf5" % (self._model_path, self._epoch))
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
        
        # load json and create model
        with open("%smodel.json" % (model_path), "r") as json_file:
            self._model = model_from_json(json_file.read())
        
        # load weights into new model
        self._model.load_weights("%sweights.%05d.hdf5" % (model_path, epoch))
        self._epoch = epoch
        self._model_path = model_path
        
        if self._verbose:
            print("Loaded model from disk")
    
    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)
    
    def predict_on_batch(self, *args, **kwargs):
        return self._model.predict_on_batch(*args, **kwargs)
    
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
        # initial definition of the sequential model
        self._inner_model = keras.Sequential(name="dl4aed")
            
    def _add_input(self):
        # for sequences (you could also add a "None" infront and get a sequence predictor)
        self._inner_model.add(Input(shape=[*self._model_parameter["input_dimensions"]], dtype=self._training_parameter["calculation_dtype"]))
        
        # TODO: think how to properly normalize the layer, depending on the input (rather do it after it was read and not augmented)
        #norm_layer = preprocessing.Normalization()
        #norm_layer.adapt(train_dataset.map(lambda x, _: x))
        #self._model.add(norm_layer)
        pass
    
    def _add_core(self):
        self._inner_model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
        self._inner_model.add(tf.keras.layers.MaxPool2D((2,2)))
        self._inner_model.add(tf.keras.layers.Dropout(0.5))

        self._inner_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self._inner_model.add(tf.keras.layers.GlobalMaxPool2D())
        
        self._inner_model.add(tf.keras.layers.Dense(self._model_parameter["embedding_dimensions"], activation="sigmoid"))
        
    def _add_output(self):
        # Define the tensors for the two input images/sequences
        left_input = tf.keras.Input(shape=[*self._model_parameter["input_dimensions"]], dtype=self._training_parameter["calculation_dtype"])
        right_input = tf.keras.Input(shape=[*self._model_parameter["input_dimensions"]])

        # Generate the encodings (feature vectors) for the two images
        encoded_l = self._inner_model(left_input)
        encoded_r = self._inner_model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = tf.keras.layers.Lambda(lambda input_: tf.keras.backend.abs(input_[0] - input_[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(self._model_parameter["output_dimensions"], activation='sigmoid')(L1_distance)

        # Connect the inputs with the outputs
        self._model = tf.keras.models.Model(inputs=[left_input, right_input], outputs=prediction)

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
        optimizer_config = { "class_name" : "adam", 
                             "config" : { "learning_rate" : self._training_parameter["learning_rate"],
                                          **self._training_parameter["optimizer_parameter"],
                                          **optimizer_clip_parameter
                                        }
                            }
        
        self._optimizer = keras.optimizers.get(optimizer_config)
        
    def loss(self, y_true, y_pred):
        # calc mean over the batch dimension
        return K.mean(self._calc_loss(y_true, y_pred), axis=0, keepdims=True)
        
    def metrics(self, y_true, y_pred):
        # return <some metrics>
        return []
    
    def _calc_loss(self, labels, predictions):
        #return <loss for each and every example>
        bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE,
                    name='binary_crossentropy')
        return bce(labels, predictions, sample_weight=None)