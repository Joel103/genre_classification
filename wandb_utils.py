'''
Usage suggestion:

    from wandb_utils import WandbWrapper
    wandb_wrapper = WandbWrapper(config)
    config = wandb_wrapper.get_config()
'''

import wandb
from matplotlib import pyplot as plt

class WandbWrapper():
    """
    A wrapper for easier W&B usage.

    ...

    Attributes
    ----------
    config : dict
        Python dictionary which holds the whole configuration of this programm
        
    Methods
    -------
    get_config():
        Returns the config used with W&B.
        
    get_callback(**kwargs):
        Returns the W&B callback which syncs metrics during the training.
        
    post_audio(y_orig, y_augm, rate, tag="Data augmentation example"):
        Uploads two audio samples to W&B.
        
    post_images(y_true, y_pred, tag="Reconstruction example"):
        Uploads two image samples to W&B.
    
    post_image(self, image, caption="", tag="Reconstruction example"):
        Uploads one image sample to W&B.
        
    post_confusion_matrix(ground_truth, predictions, class_names, tag="Confusion matrix"):
        Creates a confusion matrix (for multi-class, not multi-label problems) and uploads the image to W&B.
        
    post_pr_curve(ground_truth, predictions, class_names, classes_to_plot=None, tag="Precision-Recall"):
        Creates the PR(precision-recall) curves (for multi-class, not multi-label problems) and uploads the image to W&B.
        
    post_roc_curve(ground_truth, predictions, class_names, classes_to_plot=None, tag="ROC"):
        Creates the ROC(receiver operating characteristic) curves (for multi-class, not multi-label problems) and uploads the image to W&B.
        
    post_plt_image(data, prediction=None, title="", tag="", **imshow_kwargs):
        Uploads one or two image samples to W&B. In the case of two we put them on top of each other.
    
    post_plt_histogram(data, prediction=None, title="", tag="", **hist_kwargs):
        Uploads one or two histograms to W&B. In the case of two we make an overlay in one plot.
    """

    def __init__(self, config, **kwargs):
        self.config = config
        wandb.init(config=config)

    def get_config(self):
        '''
        Returns the config used with W&B.
        
                Returns:
                        wandb.config (dict): Dictionary with possible nested structures full of parameters for training
        '''
        return wandb.config

    def get_callback(self, **kwargs):
        '''
        Returns the W&B callback which syncs metrics during the training.

                Parameters:
                        **kwargs (dict): Dictionary of possible confuiguration changes to the callback

                Returns:
                        WandbCallback (tf.keras.callbacks.Callback): Callback which should be used during training for model tracking
        '''
        from wandb.keras import WandbCallback
        return WandbCallback(**kwargs)

    def post_audio(self, y_orig, y_augm, rate, tag="Data augmentation example"):
        '''
        Uploads two audio samples to W&B.

                Parameters:
                        y_orig (numpy.ndarray): The original track
                        y_augm (numpy.ndarray): The augmented track
                        rate (int): The sample rate of the audio track for correct playback speed
                        tag (str): The tag under which the samples should be visible (you could also use track name)
        '''
        wandb.log({tag: [ \
            wandb.Audio(y_orig, caption="Original", sample_rate=rate), \
            wandb.Audio(y_augm, caption="Augmented", sample_rate=rate), \
        ]})
    """
    def post_images(self, y_true, y_pred, tag="Reconstruction example"):
        '''
        Uploads two image samples to W&B.

                Parameters:
                        y_true (numpy.ndarray): The original image with or without augmentations
                        y_pred (numpy.ndarray): The predicted/reconstructed image from the autoencoder
                        tag (str): The tag under which the samples should be visible (you could also use track name)
        '''
        wandb.log({tag: [ \
            wandb.Image(y_true, caption="Original"), \
            wandb.Image(y_pred, caption="Prediction"), \
        ]})
    """
    """
    def post_image(self, image, caption="", tag="Image example"):
        '''
        Uploads one image sample to W&B.

                Parameters:
                        image (numpy.ndarray): The image
                        tag (str): The tag under which the sample should be visible (you could also use track name)
        '''
        wandb.log({tag: wandb.Image(image, caption=caption) })
    """
    def post_confusion_matrix(self, ground_truth, predictions, class_names, tag="Confusion matrix"):
        '''
        Creates a confusion matrix (for multi-class, not multi-label problems) and uploads the image to W&B.

                Parameters:
                        ground_truth (numpy.ndarray): An 2D array of labels
                        predictions (numpy.ndarray): An 2D array of predictions
                        class_names (list): The class names as strings
                        tag (str): The tag under which the plot should be visible

        '''
        wandb.log({tag: [ \
                   wandb.plot.confusion_matrix(predictions, ground_truth, class_names), \
        ]})

    def post_pr_curve(self, ground_truth, predictions, class_names, classes_to_plot=None, tag="Precision-Recall"):
        '''
        Creates the PR(precision-recall) curves (for multi-class, not multi-label problems) and uploads the image to W&B.

                Parameters:
                        ground_truth (numpy.ndarray): An 2D array of labels
                        predictions (numpy.ndarray): An 2D array of predictions
                        class_names (list): The class names as strings
                        classes_to_plot (list): Select a subset of all classes
                        tag (str): The tag under which the plot should be visible
        '''
        wandb.log({tag: [ \
            wandb.plot.pr_curve(ground_truth, predictions, labels=class_names, classes_to_plot=classes_to_plot), \
        ]})

    def post_roc_curve(self, ground_truth, predictions, class_names, classes_to_plot=None, tag="ROC"):
        '''
        Creates the ROC(receiver operating characteristic) curves (for multi-class, not multi-label problems) and uploads the image to W&B.

                Parameters:
                        ground_truth (numpy.ndarray): An 2D array of labels
                        predictions (numpy.ndarray): An 2D array of predictions
                        class_names (list): The class names as strings
                        classes_to_plot (list): Select a subset of all classes
                        tag (str): The tag under which the plot should be visible
        '''
        wandb.log({tag: [ \
            wandb.plot.roc_curve(ground_truth, predictions, labels=class_names, classes_to_plot=None), \
        ]})

    def post_plt_image(self, data, prediction=None, title="", tag="", **imshow_kwargs):
        '''
        Uploads one or two image samples to W&B. In the case of two we put them on top of each other.

                Parameters:
                        data (numpy.ndarray): The original image with or without augmentations
                        prediction (optional: numpy.ndarray): The predicted/reconstructed image from the autoencoder
                        title (str): Plot title
                        tag (str): The tag under which the samples should be visible (you could also use track name)
                        **imshow_kwargs (keywords/dict): all other keywords go directly to the imshow call
        '''
        fig, (ax0) = plt.subplots(1, figsize=(7, 5), dpi=100)
        if not prediction is None:
            plt.close()
            fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12, 5), dpi=100)
        im = ax0.imshow(data, interpolation=None, aspect="auto", **imshow_kwargs)
        if not prediction is None:
            im = ax1.imshow(prediction, interpolation=None, aspect="auto", **imshow_kwargs)
            fig.colorbar(im, ax=(ax0,ax1))
        else:
            fig.colorbar(im, ax=ax0)
        plt.suptitle(title)
        image = wandb.Image(plt)
        plt.close()
        return image
    
    def post_plt_histogram(self, data, prediction=None, title="", tag="", **hist_kwargs):
        '''
        Uploads one or two histograms to W&B. In the case of two we make an overlay in one plot.

                Parameters:
                        data (numpy.ndarray): The original image with or without augmentations
                        prediction (optional: numpy.ndarray): The predicted/reconstructed image from the autoencoder
                        title (str): Plot title
                        tag (str): The tag under which the samples should be visible (you could also use track name)
                        **hist_kwargs (keywords/dict): all other keywords go directly to the hist call
        '''
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        plt.hist(data.ravel(), facecolor='r', label="Image", **hist_kwargs)
        if not prediction is None:
            plt.hist(prediction.ravel(), facecolor='g', label="Prediction", **hist_kwargs)
        plt.legend()
        plt.title(title)
        plt.tight_layout(pad=1.2, h_pad=0.2, w_pad=0.2)
        image = wandb.Image(plt)
        plt.close()
        return image