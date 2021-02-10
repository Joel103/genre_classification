import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import logging

from prep_utils import *

import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

assert tf.__version__ >= '2.4.0'

class Preprocessor():
    '''
    <insert docstring>
    '''
    def __init__(self, config):
        self._config = config
        self.dataset = None
        self.noise_dataset = None
        self.ready = False
        self.logger = None
        self.noisy_samples = None
        
        self.datasets = { key : None for key in self.available_modi}
        
    def get_config(self):
        return self._config
    
    def set_config(self, update_dict):
        self._config.update(update_dict)
        return self._config
    
    @tf.autograph.experimental.do_not_convert
    def load_data(self, download=False, data_dir="tensorflow_datasets"):
        # tbd - data has to be downloaded first 
        dataset, self.ds_info = tfds.load(self._config['data_root'],
                                               with_info=True,
                                               data_dir=data_dir,
                                               download=download,
                                               split=['train'])
        dataset = dataset[0]
        dataset_size = dataset.cardinality().numpy()
        train_size = int(self._config['train_size'] * dataset_size)
        val_size = int(self._config['val_size'] * dataset_size)
        test_size = int(self._config['test_size'] * dataset_size)
        
        # split data
        dataset = dataset.shuffle(dataset_size) # is it smart to do it this way?
        self.datasets["train"] = dataset.take(train_size)
        self.datasets["test"] = dataset.skip(train_size)
        self.datasets["val"] = self.datasets["test"].skip(test_size)
        self.datasets["test"] = self.datasets["test"].take(test_size)

        meta = pd.read_csv(self._config['noise_path'], index_col='fname')
        labels = meta.label
        
        noise_classes = ['Meow', 'Cough', 'Computer_keyboard', 'Telephone',
                         'Keys_jangling', 'Knock', 'Microwave_oven', 'Finger_snapping',
                         'Bark', 'Laughter', 'Drawer_open_or_close']
        noise_data = meta.loc[meta['label'].isin(noise_classes)].index.values

        noise_ds2 = tf.data.Dataset.from_tensor_slices({'noise': noise_data,
                                                        'label': [0]*len(noise_data)}).repeat()

        self.datasets['noise'] = noise_ds2.map(lambda x: get_waveform(x,
                                                                  self._config['noise_root'],
                                                                  self._config['sample_rate']),
                                           num_parallel_calls=AUTOTUNE)
        for mode in self.available_modi:
            self.datasets[mode].cache().shuffle(self._config['shuffle_buffer_size'])
            self.offline_preprocessing(mode)

        if self.logger is not None:
            self.logger.info('dataset loaded')
    
    def create_logger(self, filepath='preprocessing.log', handler='file'):
        assert handler in ['file', 'stream'], 'handler can either be "file" or "stream"'
        logging.getLogger().handlers.clear()
        # create logger
        self.logger = logging.getLogger('preprocessing_logger')
        self.logger.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.FileHandler(filepath)
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)
        
        self.logger.propagate = False
        
        return self.logger
    
    def launch_trainval_pipeline(self, mode='train'):
        assert self.logger is not None
        assert mode in ["train", "val"]
        
        self.logger.info(f'started preprocessing and augmenting {mode} data')
        
        # zip noise dataset
        ds = tf.data.Dataset.zip((self.datasets[mode], self.datasets["noise"]))
        
        # merge features into one featureDict
        ds = ds.map(lambda x, y: wrapper_merge_features(x, y), num_parallel_calls=AUTOTUNE)

        # make to length which the network understands (common_divider)
        ds = ds.map(lambda x: wrapper_shape_to_proper_length(x, self._config["common_divider"], clip=True), num_parallel_calls=AUTOTUNE)
        
        ds = ds.map(lambda x: wrapper_roll(x, self._config['roll_val']), num_parallel_calls=AUTOTUNE) # do we wanna roll? (yes)
        ds = ds.map(lambda x: wrapper_mix_noise(x, SNR=self._config['SNR']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mask(x, self._config['freq_mask'], self._config['time_mask'], self._config['param_db'], db=0), num_parallel_calls=AUTOTUNE)
        
        # extract tensors
        ds = ds.map(lambda x: wrapper_dict2tensor(x, features=['mel', 'mel']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: wrapper_expand_both_dims(x, y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self._config['batch_size']).prefetch(AUTOTUNE)

        self.datasets[mode] = ds
        self.logger.info(f'done preprocessing and augmenting {mode} data')

    def launch_test_pipeline(self):
        
        self.logger.info('started preprocessing and augmenting test data')
        
        ds = self.datasets["test"]
        ds = ds.map(lambda x: wrapper_shape_to_proper_length(x, self._config["common_divider"], clip=True, testing=True), num_parallel_calls=AUTOTUNE)

        # extract tensors
        ds = ds.map(lambda x: wrapper_dict2tensor(x, features=['mel','label']), num_parallel_calls=AUTOTUNE)
        def wrapper_expand_dims(x, y):
            return (tf.expand_dims(x, -1), y)
        ds = ds.map(lambda x, y: wrapper_expand_dims(x,y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self._config['batch_size']).prefetch(AUTOTUNE)
        
        self.test_dataset = ds
        
        self.logger.info('done preprocessing and augmenting test data')
    
    def preprocess_wav(self, ds):
        ds = ds.map(lambda x: wrapper_cast_offline(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_rescale(x), num_parallel_calls=AUTOTUNE)
        '''
        ds = ds.map(lambda x: wrapper_trim(x, self._config['epsilon']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_change_pitch(x, self._config['shift_val'], self._config['bins_per_octave'], self._config['sample_rate']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_fade(x, self._config['fade']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)        
        '''
        return ds

    def to_log_mel(self, ds):
        ds = ds.map(lambda x: wrapper_gpu(x,
                                            self._config['nfft'],
                                            self._config['window'],
                                            self._config['stride'],
                                          self._config['sample_rate'], self._config['mels'],
                                          self._config['fmin_mels'], self._config['fmax_mels'],
                                          self._config['top_db'], db=0, logger=self.logger),
                    num_parallel_calls=AUTOTUNE)
        
        ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_extract_shape(x), num_parallel_calls=AUTOTUNE)
        
        return ds

    def offline_preprocessing(self, mode='val'):
        '''
        create spectrograms to spare processing time due to fourier transforms
        mode in self.available_modi (['noise', 'train', 'val', 'test'])
        '''
        assert mode in self.available_modi
        
        ds = self.datasets[mode]
        
        # wav-level
        ds = self.preprocess_wav(ds)
        # spect-level
        ds = self.to_log_mel(ds)
        # adding shape information
        ds = ds.map(lambda x: wrapper_extract_shape(x), num_parallel_calls=AUTOTUNE)

        self.datasets[mode] = ds
        
        self.logger.info('done preprocessing and augmenting data')

    @property
    def available_modi(self):
        return ['train', 'val', 'test', 'noise']
    
    @property
    def train_ds(self):
        return self.datasets['train']
    
    @property
    def val_ds(self):
        return self.datasets['val']
    
    @property
    def test_ds(self):
        return self.datasets['test']
    
    @property
    def noise_ds(self):
        return self.datasets['noise']