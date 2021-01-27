import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import librosa

import logging

from prep_utils import *

tfd = tfp.distributions
AUTOTUNE = tf.data.AUTOTUNE

assert tf.__version__ == '2.4.0'

'''
TODO:
 * Wassim: train/val/test split of data (tfds.load --> split) 
 * Wassim: noise dataset under git version control (LFS)
 * Joel: video about melspectrogram
 * Nico: correct paths
 * All: MFCC needed?
 * All: Power of spectrogram before melscale?
'''

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
        
    def get_config(self):
        return self._config
    
    def set_config(self, update_dict):
        self._config.update(update_dict)
        return self._config
    
    def load_data(self, download=False):
        # tbd - data has to be downloaded first 
        self.dataset, self.ds_info = tfds.load(self._config['data_root'],
                                               with_info=True,
                                               download=download,
                                               split=['train'])
        self.dataset = self.dataset[0]
        
        meta = pd.read_csv(self._config['noise_path'], index_col='fname')
        labels = meta.label
        
        noise_classes = ['Meow', 'Cough', 'Computer_keyboard', 'Telephone',
                         'Keys_jangling', 'Knock', 'Microwave_oven', 'Finger_snapping',
                         'Bark', 'Laughter', 'Drawer_open_or_close']
        noise_data = meta.loc[meta['label'].isin(noise_classes)].index.values
        random.shuffle(noise_data)
        size = self.dataset.cardinality()
        extended_noise = noise_data
        while len(extended_noise) < size:
            extended_noise = np.concatenate([extended_noise, noise_data])
            
        noise_ds2 = tf.data.Dataset.from_tensor_slices({'noise': extended_noise,
                                                        'label': [0]*len(extended_noise)})

        self.noise_dataset = noise_ds2.map(lambda x: get_waveform(x,
                                                                  self._config['noise_root'],
                                                                  self._config['sample_rate']),
                                           num_parallel_calls=AUTOTUNE)  
        
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
    
    def launch_pipeline(self, extract_noise_wav=False):
        assert self.dataset is not None
        assert self.noise_dataset is not None
        assert self.logger is not None
        
        self.logger.info('started preprocessing and augmenting data')
        
        # zip noise dataset
        ds = tf.data.Dataset.zip((self.dataset, self.noise_dataset))
        # merge features into one featureDict
        ds = ds.map(lambda x, y: wrapper_merge_features(x, y), num_parallel_calls=AUTOTUNE)

        # wav-level
        ds = ds.map(lambda x: wrapper_cast(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_cut_15(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_change_pitch(x, self._config['shift_val'], self._config['bins_per_octave']),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_trim(x, self._config['epsilon']), num_parallel_calls=AUTOTUNE) # do we wanna trim?
        ds = ds.map(lambda x: wrapper_fade(x, self._config['fade']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_pad_noise(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mix_noise(x, SNR=self._config['SNR']), num_parallel_calls=AUTOTUNE)
        # spect-level
        ds = ds.map(lambda x: wrapper_spect(x, self._config['nfft'], self._config['window'], self._config['stride']),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mel(x,
                                          self._config['sample_rate'], self._config['mels'],
                                          self._config['fmin_mels'], self._config['fmax_mels'],
                                          self._config['top_db'], db=0),
                    num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
        ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mfcc(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_roll(x, self._config['roll_val']),
                    num_parallel_calls=AUTOTUNE) # do we wanna roll? (yes)
        ds = ds.map(lambda x: wrapper_mask(x, self._config['freq_mask'],
                                           self._config['time_mask'], self._config['param_db'],
                                           db=0),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=self._config['shuffle_batch_size'])
        
        self.dataset = ds
        
        self.logger.info('done preprocessing and augmenting data')
    
    def getter(self):
        return self.dataset