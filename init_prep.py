config = {}
config['data_root'] = 'gtzan'
config['noise_path'] = 'C:/Users/BOUBAW/tensorflow_datasets/fsd/meta/test_post_competition_scoring_clips.csv'
config['sample_rate'] = 22050
config['nfft'] = 512
config['window'] = 512
config['stride'] = 256
config['mels'] = 128
config['fmin_mels'] = 0
config['fmax_mels'] = 8000
config['time_mask'] = 10
config['freq_mask'] = 10
config['noise_threshold'] = 1 # add noise to only 0.3
config['beta'] = 0.5 # noise strength when mixing mel spectrograms
config['SNR'] = 5
config['noise_root'] = 'C:/Users/BOUBAW/tensorflow_datasets/fsd/1.0.0'
config['shuffle_batch_size'] = 64


y = Preprocessor(config=config)
y.create_logger()
y.load_data()

y.set_config({'fade': 10000,
              'epsilon': 0.1,
              'roll_val': 15,
              'top_db': 80,
              'shift_val': 3,
              'bins_per_octave': 12,
              'param_db': 10,
              'train_size': 0.7,
              'val_size': 0.2,
              'test_size': 0.1,
              'noisy_samples': 5})
