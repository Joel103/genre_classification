import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import librosa

import logging

from prep_utils import *
from utils import wrapper_serialize, save_dataset, load_dataset

import tensorflow_probability as tfp
import tensorflow_datasets as tfds

tfd = tfp.distributions
AUTOTUNE = tf.data.AUTOTUNE

assert tf.__version__ == '2.4.0'

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
        
    def get_config(self):
        return self._config
    
    def set_config(self, update_dict):
        self._config.update(update_dict)
        return self._config
    
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
        self.train_dataset = dataset.take(train_size)
        self.test_dataset = dataset.skip(train_size)
        self.val_dataset = self.test_dataset.skip(val_size)
        self.test_dataset = self.test_dataset.take(test_size)

        
        meta = pd.read_csv(self._config['noise_path'], index_col='fname')
        labels = meta.label
        
        noise_classes = ['Meow', 'Cough', 'Computer_keyboard', 'Telephone',
                         'Keys_jangling', 'Knock', 'Microwave_oven', 'Finger_snapping',
                         'Bark', 'Laughter', 'Drawer_open_or_close']
        noise_data = meta.loc[meta['label'].isin(noise_classes)].index.values
        random.shuffle(noise_data)
        size = self.train_dataset.cardinality()
        extended_noise = noise_data
        while len(extended_noise) < size:
            extended_noise = np.concatenate([extended_noise, noise_data])
            
        noise_ds2 = tf.data.Dataset.from_tensor_slices({'noise': extended_noise,
                                                        'label': [0]*len(extended_noise)})

        self.noise_dataset = noise_ds2.map(lambda x: get_waveform(x,
                                                                  self._config['noise_root'],
                                                                  self._config['sample_rate']),
                                           num_parallel_calls=AUTOTUNE).cache()
        
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
    
    def launch_trainval_pipeline(self, extract_noise_wav=False, mode='train'):
        assert self.train_dataset is not None
        assert self.noise_dataset is not None
        assert self.logger is not None
        
        self.logger.info(f'started preprocessing and augmenting {mode} data')
        
        if mode=='train':
            ds = self.train_dataset
        elif mode=='val':
            ds = self.val_dataset
        else:
            raise Excpetion('wrong pipeline. For test pipeline, call launch_test_pipeline.')
        
        # zip noise dataset
        ds = tf.data.Dataset.zip((ds, self.noise_dataset))
        # merge features into one featureDict
        ds = ds.map(lambda x, y: wrapper_merge_features(x, y), num_parallel_calls=AUTOTUNE)

        # wav-level
        ds = ds.map(lambda x: wrapper_cast(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_cut_15(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_change_pitch(x, self._config['shift_val'], self._config['bins_per_octave'], self._config['sample_rate']),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_trim(x, self._config['epsilon']), num_parallel_calls=AUTOTUNE) # do we wanna trim?
        ds = ds.map(lambda x: wrapper_fade(x, self._config['fade']), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_pad_noise(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mix_noise(x, SNR=self._config['SNR']), num_parallel_calls=AUTOTUNE)
        
        ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)
        
        # extract noisy samples to hear
        if mode=='train':
            self.noisy_samples = ds.take(self._config['noisy_samples'])
        
        # spect-level
        ds = ds.map(lambda x: wrapper_spect(x, self._config['nfft'], self._config['window'], self._config['stride']),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mel(x,
                                          self._config['sample_rate'], self._config['mels'],
                                          self._config['fmin_mels'], self._config['fmax_mels'],
                                          self._config['top_db'], db=0),
                    num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
        ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
        #ds = ds.map(lambda x: wrapper_mfcc(x), num_parallel_calls=AUTOTUNE)
        #ds = ds.map(lambda x: wrapper_roll(x, self._config['roll_val']), num_parallel_calls=AUTOTUNE) # do we wanna roll? (yes)
        ds = ds.map(lambda x: wrapper_mask(x, self._config['freq_mask'], self._config['time_mask'], \
                                           self._config['param_db'], db=0), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(buffer_size=self._config['shuffle_batch_size'])
        
        # extract tensors
        ds = ds.map(lambda x: wrapper_dict2tensor(x, features=['mel','label']),
                    num_parallel_calls=AUTOTUNE)

        
        if mode=='train':
            self.train_dataset = ds
        elif mode=='val':
            self.val_dataset = ds
        
        self.logger.info(f'done preprocessing and augmenting {mode} data')


    def launch_test_pipeline(self, extract_noise_wav=False):
        
        self.logger.info('started preprocessing and augmenting test data')
        
        ds = self.test_dataset
        
        ds = ds.map(lambda x: wrapper_cast(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_cut_15(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_spect(x, self._config['nfft'], self._config['window'], self._config['stride']),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mel(x,
                                          self._config['sample_rate'], self._config['mels'],
                                          self._config['fmin_mels'], self._config['fmax_mels'],
                                          self._config['top_db'], db=0),
                    num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
        ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: wrapper_mfcc(x), num_parallel_calls=AUTOTUNE)
        
        # extract tensors
        ds = ds.map(lambda x: wrapper_dict2tensor(x, features=['mel','label']),
                    num_parallel_calls=AUTOTUNE)
        
        self.test_dataset = ds
        
        self.logger.info('done preprocessing and augmenting test data')
        
    def offline_preprocessing(self, mode=False):
        '''
        create spectrograms to spare processing time due to fourier transforms
        mode in ['noise', 'train', 'val', 'test']
        '''
        if mode=='noise':
            ds = self.noise_dataset
            # wav-level
            ds = ds.map(lambda x: wrapper_cast_offline(x, noise=True), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)
            # spect-level
            ds = ds.map(lambda x: wrapper_spect(x,
                                                self._config['nfft'],
                                                self._config['window'],
                                                self._config['stride'],
                                                self.logger),
                        num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_mel(x,
                                              self._config['sample_rate'], self._config['mels'],
                                              self._config['fmin_mels'], self._config['fmax_mels'],
                                              self._config['top_db'], db=0, logger=self.logger),
                        num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
            ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
            
            self.noise_dataset = ds
        
        elif mode=='test':
            ds = self.test_dataset
            
            ds = ds.map(lambda x: wrapper_cast_offline(x), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_cut_15(x), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_spect(x, self._config['nfft'], self._config['window'], self._config['stride'], logger=self.logger),
                        num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_mel(x,
                                              self._config['sample_rate'], self._config['mels'],
                                              self._config['fmin_mels'], self._config['fmax_mels'],
                                              self._config['top_db'], db=0, logger=self.logger),
                        num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
            ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)

            self.test_dataset = ds
        
        else:
            if mode=='train':
                ds = self.train_dataset
            else:
                ds = self.val_dataset
            
            # wav-level
            ds = ds.map(lambda x: wrapper_cast_offline(x), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_cut_15(x), num_parallel_calls=AUTOTUNE)
            # no trim and no pitch shift
            ds = ds.map(lambda x: wrapper_fade(x, self._config['fade']), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_normalize(x), num_parallel_calls=AUTOTUNE)
            
            # spect-level
            ds = ds.map(lambda x: wrapper_spect(x, self._config['nfft'], self._config['window'], self._config['stride']),
                        num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x: wrapper_mel(x,
                                              self._config['sample_rate'], self._config['mels'],
                                              self._config['fmin_mels'], self._config['fmax_mels'],
                                              self._config['top_db'], db=0),
                        num_parallel_calls=AUTOTUNE) # Convert to mel-spectrogram
            ds = ds.map(lambda x: wrapper_log_mel(x), num_parallel_calls=AUTOTUNE)
            
            if mode=='train':
                self.train_dataset = ds
            else:
                self.val_dataset = ds

        
        self.logger.info('done preprocessing and augmenting data')
    
    def save_mels(self, skip=[]):
        for mode, ds in zip(['train', 'val', 'noise', 'test'], [self.train_dataset, self.val_dataset, self.noise_dataset, self.test_dataset]):
            if mode in skip:
                continue
            mels_ds = ds.map(lambda x: wrapper_dict2tensor(x, ['mel']))
            
            if mode=='noise':
                labels_ds = ds.map(lambda x: wrapper_dict2tensor(x, ['noise_label']))
            else:
                labels_ds = ds.map(lambda x: wrapper_dict2tensor(x, ['label']))

            _ = save_dataset(mels_ds, f'./{mode}/mels.record')
            _ = save_dataset(labels_ds, f'./{mode}/labels.record')
            
            self.logger.info(f'done saving data for {mode}')
        
            
            
    
    
    
    @property
    def train_ds(self):
        return self.dataset