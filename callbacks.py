import tensorflow as tf

class IncreaseEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self, network):
        self.network = network
    def on_epoch_end(self, epoch, logs=None):
        # Since Keras Progbar starts counting with 1, I have to add here 1 
        self.network.epoch = epoch+1

# Tensorflow Keras ModelCheckpoint argument 'period' is deprecated
# Therefore, I'm doing it on my own
class SaveEveryNthEpochCustom(tf.keras.callbacks.Callback):
    def __init__(self, network, save_steps):
        self.network = network
        self.save_steps = save_steps
    def on_epoch_end(self, epoch, logs=None):
        if self.network.epoch % self.save_steps == 0:
            self.network.save()
