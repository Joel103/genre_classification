import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import librosa

import time

#============================== PREPROCESSING ==============================

def wrapper_cast(x):
    x['audio'] = tf.cast(x['audio'], tf.float32)
    try:
        x['noise_wav'] = tf.cast(x['noise_wav'], tf.float32)
    except KeyError:
        x['input'] = x['audio']
    return x

def wrapper_cast_offline(x):
    try:
        x['input'] = tf.cast(x['noise_wav'], tf.float32)
    except KeyError:
        x['input'] = tf.cast(x['audio'], tf.float32)
    return x

def wrapper_normalize(x):
    # normalize whole sample in a range between 0 and 1
    x['mel'] -= tf.math.reduce_min(x['mel'])
    x['mel'] /= tf.math.reduce_max(x['mel'])
    return x

def wrapper_rescale(x):
    # normalize whole sample in a range between 0 and 1
    x['input'] /= tf.math.reduce_max(x['input'])
    return x

def wrapper_gpu(x, nfft, window, stride, sample_rate, mels, fmin_mels, fmax_mels, top_db, db, logger):
    with tf.device("/gpu:0"):
        x = wrapper_spect(x, nfft, window, stride, logger)
        x = wrapper_mel(x, sample_rate, mels, fmin_mels, fmax_mels, top_db, db, logger)
        return x
    
def wrapper_spect(x, nfft, window, stride, logger=None):
    
    start_time = time.time()

    x['spectrogram'] = tfio.experimental.audio.spectrogram(x['input'],
                                                           nfft=nfft,
                                                           window=window,
                                                           stride=stride)
        
    if logger is not None:
        logger.info(f'computing spectrogram took {np.round(time.time() - start_time, 2)} s')

    x.pop('input')
    return x

def wrapper_mel(x, sample_rate, mels, fmin_mels, fmax_mels, top_db, db=False, logger=None):
    start_time = time.time()
    x['mel'] = tfio.experimental.audio.melscale(x['spectrogram'],
                                                rate=sample_rate,
                                                mels=mels,
                                                fmin=fmin_mels,
                                                fmax=fmax_mels)
    if db: #to be implemented with noise
        x['db_mel'] = tfio.experimental.audio.dbscale(x['mel'], top_db=top_db)
        
    if logger is not None:
        logger.info(f'computing mel-spectrogram took {np.round(time.time() - start_time, 2)} s')
    
    x.pop('spectrogram')
    return x
'''
def cut_15(signal):
    start = np.random.randint(0, int(len(signal)/2))
    signal = signal[start:start+int(len(signal)/2)]
    return signal

def wrapper_cut_15(x):
    out = tf.py_function(cut_15, [x['input']], [tf.float32])
    x['input'] = tf.squeeze(out)
    return x
'''
#============================== AUGMENTATION ==============================

def wrapper_fade(x, fade):
    x['input'] = tfio.experimental.audio.fade(x['input'], fade_in=fade, fade_out=fade, mode="logarithmic")
    return x

def wrapper_trim(x, epsilon):
    position = tfio.experimental.audio.trim(x['audio'], axis=0, epsilon=epsilon)
    start = position[0]
    stop = position[1]
    x['audio'] = x['audio'][start:stop]
    return x

def wrapper_mask(x, freq_mask, time_mask, param_db, db=False):
    # freq masking
    x['mel'] = tfio.experimental.audio.freq_mask(x['mel'],
                                                 param=freq_mask)
    # Time masking
    x['mel'] = tfio.experimental.audio.time_mask(x['mel'],
                                                 param=time_mask)
    if db:
        x['db_mel'] = tfio.experimental.audio.freq_mask(x['db_mel'], param=param_db)
        x['db_mel'] = tfio.experimental.audio.time_mask(x['db_mel'], param=param_db)
    return x

def wrapper_roll(x, roll_val):
    roll_tensor = tf.random.uniform((), minval=-roll_val, maxval=roll_val, dtype=tf.dtypes.int32)
    x['mel'] = tf.roll(x['mel'], roll_tensor, axis=0)
    x['mel_noise'] = tf.roll(x['mel_noise'], roll_tensor, axis=0)
    return x

def get_noise_from_sound(signal, noise, SNR):
    # current RMS of signal
    centered_signal = signal-tf.reduce_mean(signal)
    RMS_s = tf.sqrt(tf.reduce_mean(tf.square(centered_signal)))
    
    # current RMS of noise
    centered_noise = noise-tf.reduce_mean(noise)
    RMS_n_current = tf.sqrt(tf.reduce_mean(tf.square(centered_noise)))
    
    # scalar
    RMS_n = SNR * (RMS_s / RMS_n_current)
    
    noise /= RMS_n
    return noise

def wrapper_mix_noise(x, SNR):
    out = get_noise_from_sound(x['mel'], x['mel_noise'], SNR)
    x['mel'] += tf.squeeze(out)
    return x

def wrapper_merge_features(ds, ds_noise):
    ds.update({"mel_noise": ds_noise["mel"], "label_noise": ds_noise["label"], "shape_noise": ds_noise["shape"]})
    return ds

def wrapper_shape_to_proper_length(x, common_divider, clip=True, testing=False):
    # TODO: adapt to be dynamic about how long either signal or noise is (current assumption is: signal_length >= noise_length)
    
    # get shapes
    signal_shape = x['shape']
    if clip:
        pad_length = signal_shape[0] - tf.math.mod(signal_shape[0], common_divider)
        x["mel"] = x["mel"][:pad_length]
    else:
        # calc desired sequence length
        pad_length = signal_shape[0] + common_divider - tf.math.mod(signal_shape[0], common_divider)

        # create padding
        signal_zeros = tf.zeros((pad_length - signal_shape[0], signal_shape[1]), tf.float32)
        x["mel"] = tf.concat([x["mel"], signal_zeros], axis=0)

    if not testing:
        # pad
        noise_shape = x['shape_noise']
        noise_zeros = tf.zeros((pad_length - noise_shape[0], noise_shape[1]), tf.float32)    
        x["mel_noise"] = tf.concat([x["mel_noise"], noise_zeros], axis=0)

    return x

def wrapper_log_mel(x):
    x['mel'] = tf.math.log(1 + x['mel'])
    return x

def wrapper_extract_shape(x):
    x["shape"] = tf.shape(x["mel"])
    return x

def get_waveform(x, noise_root, sample_rate):
    audio_binary = tf.io.read_file(noise_root+os.sep+x['noise'])
    audio, _ = tf.audio.decode_wav(audio_binary,
                                   desired_channels=1,
                                   desired_samples=sample_rate)
    audio = tf.squeeze(audio, axis=-1)
    
    return {'audio': audio,
            'label': tf.cast(x['label'], tf.int32),
            'rate': sample_rate}

def wrapper_dict2tensor(x, features=['mel','label']):
    return [tf.convert_to_tensor(x[feature]) for feature in features]

def wrapper_expand_both_dims(x, y):
    return (tf.expand_dims(x, -1), tf.expand_dims(y, -1))

def wrapper_pack(x):
    return {"label": tf.cast(x["label"], tf.int32), "mel":x["mel"], "shape":x["shape"]}

def pitch_shift_data(wave_data, shift_val, bins_per_octave, sample_rate):

    wave_data = wave_data.numpy()
    random_shift = np.random.randint(low=-shift_val, high=shift_val)
    wave_data = librosa.effects.pitch_shift(wave_data, sample_rate, 
                                            random_shift, bins_per_octave=bins_per_octave.numpy())
    return wave_data

def wrapper_change_pitch(x, shift_val, bins_per_octave, sample_rate):
    out = tf.py_function(pitch_shift_data, [x['audio'], shift_val, bins_per_octave, sample_rate], tf.float32)
    x['audio'] = out
    return x