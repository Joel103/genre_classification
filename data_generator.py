import tensorflow as tf
import tensorflow_io as tfio
import glob
from pathlib import Path

# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE

def download_data():
    '''
    returns filepath
    '''
    # download the dataset using this neat keras function
    download_link = 'https://gitlab.tubit.tu-berlin.de/dl4aed/dl4aed-data/raw/master/TinyUrbanSound8k/TinyUrbanSound8k.tar.gz'
    return tf.keras.utils.get_file(Path('./_data/TinyUrbanSound8k.tar.gz').resolve(),
                                    download_link,
                                    cache_subdir=Path('./_data/').resolve(),
                                    extract=True)

download_data()

folders = glob.glob('_data/TinyUrbanSound8k/*/*/')
classes = tf.constant(sorted(set([Path(f).parts[-1] for f in folders])))
num_classes = tf.shape(classes)[0]

# TODO: push those into the config
power = 2.
nfft = 2048
stride = 256
mels = 64
top_db = 80.
sr = 16000
fmin = 0.
fmax = sr/2.
wav_normalize_scale = 2**15
length = 1
image_width = tf.cast(tf.math.ceil(length*sr/stride), dtype=tf.int32).numpy()

def load_file(file_path):
    _audio = tfio.audio.AudioIOTensor(file_path, dtype=tf.dtypes.int16)
    return tf.cast(tf.squeeze(_audio.to_tensor(), axis=-1), dtype=tf.float32) / wav_normalize_scale, tf.cast(_audio.rate, dtype=tf.float32)

def folder_name_to_one_hot(file_path):
    # for example: _data/TinyUrbanSound8k/train/siren/157648-8-0-0_00.wav
    label = tf.strings.split(file_path, sep="/")[-2]
    label_idx = tf.argmax(tf.cast(tf.equal(classes, label), tf.int32))

    # get one hot encoded array
    one_hot = tf.one_hot(label_idx, num_classes, on_value=None, off_value=None, axis=None, dtype=tf.float32, name=None)
    return one_hot

def get_mel(audio):
    spectrogram = tfio.experimental.audio.spectrogram(audio, nfft=nfft, window=nfft, stride=stride)
    # due to the bad validation of tfio.experimental.audio.melscale the sr has to be a python variable
    mel_spectrograms = tfio.experimental.audio.melscale(spectrogram**power, rate=sr, mels=mels, fmin=fmin, fmax=fmax)
    return tf.transpose(tfio.experimental.audio.dbscale(mel_spectrograms, top_db=top_db))

def load_and_preprocess_data(file_path):
    # get ground truth from file_path string
    one_hot = folder_name_to_one_hot(file_path)

    # load audio data and transform
    audio, _ = load_file(file_path)
    audio = get_mel(audio)
    audio = tf.expand_dims(audio, axis=-1)

    return audio, one_hot

def train_augment(x):
    # TODO: add more augmentations
    x = tf.transpose(x[:,:,0])
    x = tfio.experimental.audio.freq_mask(x, param=10)
    x = tfio.experimental.audio.time_mask(x, param=10)
    x = tf.expand_dims(tf.transpose(x), axis=-1)
    return tf.cast(x, dtype=tf.float32)

# Probably works, eventhough sometimes the images are the same and the label is "false"
# -> i guess, there are duplicates in the sets
# TODO: Rethink if we should only compare directly if it's the same track, or if we work with a class vector and reduce it depending on the class?
# -> Remember: tf.math.reduce_all(tf.math.equal(y[0], y[0])
def extract(x, y):
    # is 50:50 always the right thing to compare either the same track or two different ones? --> parameterize 
    pred = tf.cast(tf.random.uniform(tf.cast([1],dtype=tf.int32), 
                                     minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None), dtype=tf.bool)
    def true_fn():
        return (train_augment(x[0]), train_augment(x[0])), tf.cast(True, dtype=tf.bool)
    def false_fn():
        return (train_augment(x[0]), train_augment(x[1])), tf.cast(False, dtype=tf.bool)
    return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=None)

def create_train_data(training_parameter):

    # folder with the training data
    train_files = './_data/TinyUrbanSound8k/train/*/*.wav'

    # define a dataset of file paths
    train_dataset = tf.data.Dataset.list_files(train_files)
    # run the preprocessing via map
    train_dataset = train_dataset.map(load_and_preprocess_data, num_parallel_calls=AUTOTUNE).cache()
    # shuffle the data
    train_dataset = train_dataset.shuffle(buffer_size=4000)

    # create mix/matches
    train_dataset = train_dataset.batch(2)
    train_dataset = train_dataset.map(extract, num_parallel_calls=AUTOTUNE, deterministic=None)

    # batch examples
    train_dataset = train_dataset.batch(training_parameter["batch_size"])

    # prefetch
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    return train_dataset
    
def create_test_data(training_parameter):
    # TODO: think about the difficulty of the extraction during inference
    # TODO: think about common metrics and evaluation techniques (accuracy may not be relevant anymore)

    # folder with the evaluation data
    test_files = './_data/TinyUrbanSound8k/test/*/*.wav'

    # define a dataset of file paths
    test_dataset = tf.data.Dataset.list_files(test_files)
    # run the preprocessing via map
    test_dataset = test_dataset.map(load_and_preprocess_data, num_parallel_calls=AUTOTUNE).cache()

    # TODO: with a new metric, this step might be irrelevant (we're also augmenting the test data)
    # create mix/matches
    test_dataset = test_dataset.batch(2)
    test_dataset = test_dataset.map(extract, num_parallel_calls=AUTOTUNE, deterministic=None)

    # batch examples
    test_dataset = test_dataset.batch(training_parameter["batch_size"])
    
    # prefetch
    test_dataset = test_dataset.prefetch(AUTOTUNE)
    return test_dataset