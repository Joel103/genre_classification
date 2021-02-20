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

class ReconstructImages(tf.keras.callbacks.Callback):
    def __init__(self, network, period, dataset, wandb_wrapper):
        self.network = network
        self.period = period
        self.dataset = dataset
        self.wandb_wrapper = wandb_wrapper
        self.plot_images = 5
        
    def on_epoch_end(self, epoch, logs=None):
        if self.network.epoch % self.period == 0:
            self.reconstruct_images()

    def reconstruct_images(self):
        import numpy as np
        from matplotlib import pyplot as plt
        import wandb
        images = []
        histograms = []
        for elem in self.dataset:
            batch_size = elem[0].shape[0]
            prediction = self.network.predict_on_batch(elem[0])["decoder"]
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            for index in indices[:self.plot_images]:
                x = elem[0][index][..., 0].numpy().astype(np.float32).T
                y = prediction[index][..., 0].astype(np.float32).T
                images += [self.wandb_wrapper.post_plt_image(x, y, title="Images", tag="side-by-side-images")]
                histograms += [self.wandb_wrapper.post_plt_histogram(x, y, title="Histogram", tag="overlay-histogram", alpha=0.35, bins=50)]
            break
        wandb.log({"side-by-side-images": images})
        wandb.log({"overlay-histogram": histograms})
            
class CreateEmbedding(tf.keras.callbacks.Callback):
    def __init__(self, network, period, dataset, num_classes=10):
        self.network = network
        self.period = period
        self.dataset = dataset
        self.num_classes = num_classes
        self._plotted_random_samples = 1000
        
    def on_epoch_end(self, epoch, logs=None):
        if self.network.epoch % self.period == 0:
            self.create_embedding()

    def create_embedding(self):
        import numpy as np
        import wandb
        from sklearn.manifold import TSNE
        from itertools import cycle
        from matplotlib import pyplot as plt
        from matplotlib.ticker import NullFormatter

        ''' Collect Network Embeddings '''
        collect_embeddings = []
        collect_labels = []
        for elem in self.dataset:
            for (x, y) in zip(elem[0], elem[1]["classifier"]):
                prediction = self.network.predict_embedding_on_batch(x[np.newaxis])
                collect_embeddings += [prediction]
                collect_labels += [tf.argmax(y, axis=1)]
        
        ''' Perform T-SNE '''
        embeddings = tf.concat(collect_embeddings, axis=0).numpy()
        labels = tf.concat(collect_labels, axis=0).numpy()
        X_embedded = TSNE(n_components=2).fit_transform(np.squeeze(embeddings))
        
        ''' Some Preparation For Colored Plotting '''
        collect_colors_markers = {}
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])
        markers = cycle(('o', ','))
        
        for i in range(self.num_classes):
            collect_colors_markers[i] = (next(colors), next(markers))

        ''' Scatter Plot Embeddings '''
        indices = np.random.choice(labels.shape[0], self._plotted_random_samples, replace=False)
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        # Add scatter plot
        ax = fig.add_subplot(111)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.25, s=0.1, c="gray")
        for embedding, label in zip(X_embedded[indices], labels[indices]):
            ax.scatter(embedding[0], embedding[1], alpha=0.5, c=collect_colors_markers[label][0], marker=collect_colors_markers[label][1])

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        
        # send to wandb
        wandb.log({"test embedding - label colored": wandb.Image(plt)})
        plt.close()
        return embeddings, labels