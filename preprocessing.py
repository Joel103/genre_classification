# -*- coding: utf-8 -*-

import tensorflow_io as tfio



def spectogram(input, nfft, window, stride):
  """Convert a waveform into a spectogram
    Args:
      input: An 1-D audio signal Tensor.
      nfft: Size of FFT.
      window: Size of window.
      stride: Size of hops between windows.
    Returns:
      A tensor of spectrogram
    """
  return tfio.experimental.audio.spectrogram(input, nfft=nfft, window=window, stride=stride)

def mel_spectogram(input, rate, mels, fmin, fmax):

  """
  Turn spectrogram into mel scale spectrogram
    Args:
      input: A spectrogram Tensor with shape [frames, nfft+1].
      rate: Sample rate of the audio.
      mels: Number of mel filterbanks.
      fmin: Minimum frequency. 
      fmax: Maximum frequency. 
      name: A name for the operation (optional).
    Returns:
      A tensor of mel spectrogram with shape [frames, mels].
    """
  return tfio.experimental.audio.melscale(input, rate=rate, mels=mels, fmin=fmin, fmax=fmax)

def mfcc():
  """
  Still has to be constructed
    """
