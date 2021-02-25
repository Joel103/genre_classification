import numpy as np
import matplotlib.pyplot as plt
import annoy
from annoy import AnnoyIndex
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class ANN():
    '''
    (super sorry, but I don't really have time. Shame on me.)
    '''
    
    def __init__(self, base_population_path, label_path, filename_path, experimental_size,
                 metric='angular', n_trees=10):
        self.filename_path = filename_path
        self.base_population_path = base_population_path
        self.label_path = label_path
        self.experimental_size = experimental_size
        self.metric = metric
        self.base_population = None
        self.labels = None
        self.n_trees = n_trees
        self.genre_names = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock',
                            'International', 'Electronic', 'Instrumental']
        
        self._load_population() # load files
        self.embedding_size = self.base_population.shape[1]
        
        self.annoy_index = AnnoyIndex(self.embedding_size, metric)
        
        
        # add population
        for sample_id, embedding in enumerate(self.base_population):
            self.annoy_index.add_item(sample_id, embedding)
        
        # build 
        self.annoy_index.build(n_trees)
        
    def _prepare_tsne(self):
        print('working on TSNE. Might take a while')
        self.tsne = TSNE(n_components=2, random_state=0)
        self.tsne_output = self.tsne.fit_transform(self.base_population)
        print('Oof. Done with TSNE.')
        
    def add_reindex(self, new_embeddings, add_to_base=False):
        '''
        new_embeddings has to a be an "array of arrays", more like a list of arrays if 
        you get what I mean.
        '''
        _ = self.annoy_index.unbuild()
        if not add_to_base:
            for sample_id, embedding in enumerate(new_embeddings):
                self.annoy_index.add_item(sample_id+len(self.base_population), embedding) 
            self.annoy_index.build(self.n_trees)
        else:
            raise NotImplementedError('Sorry, no time to implement this yet.')
    
    def supervised_evaluation(self, top_x=2, double_vote=True):
        '''
        see if labels assigned by voting really match closest neighbors 
        '''
        self.predictions = {}
        self.sub_predictions = []
        for embed_vec, filename, genre in zip(self.base_population, self.filenames, self.labels):
            mc = list(self.assign_label(embed_vec))
            self.sub_predictions.append(mc)
            if filename in self.predictions.keys():
                self.predictions[filename][0].extend(mc)
            else:
                self.predictions[filename] = [mc, genre]
        
        # top x sub_track classification
        self.top_x = np.sum([label in pred for label, pred in zip(self.labels, self.sub_predictions)])/len(self.labels)
        
        squeezed_labels = np.array([self.predictions[filename][1] for filename in self.predictions])
        if double_vote:
            # song-genre classification
            most_commons = np.array([Counter(self.predictions[filename][0]).most_common()[0][0] for filename in self.predictions])
            self.assigned_classes = np.sum(most_commons == squeezed_labels) /len(squeezed_labels)
            self.classification_report = classification_report(squeezed_labels, most_commons, target_names=self.genre_names)
            self.confusion_matrix = confusion_matrix(squeezed_labels, most_commons, labels=range(len(self.genre_names)))
        else:
            # sub-track-genre classification
            most_commons = np.array([Counter(knns).most_common()[0][0] for knns in self.sub_predictions])
            self.assigned_classes = np.sum(most_commons==self.labels)/len(self.labels)
            self.classification_report = classification_report(self.labels, most_commons, target_names=self.genre_names)
            self.confusion_matrix = confusion_matrix(self.labels, most_commons, labels=range(len(self.genre_names)))
            
        print(self._evaluation_report(top_x, double_vote))
        
#         unit_predictions = [pred[0] for pred in self.predictions]
        
        
    
    def _evaluation_report(self, top_x, double_vote):
        out = 'song-' if double_vote else 'sub-track-'
        out_str = f'{out}classification acc: {100 * np.round(self.assigned_classes, 2)}% \n'
        out_str += f'top_{top_x} sub-track-classification acc: {100 * np.round(self.top_x, 2)}%'
        return out_str
    
    def assign_label(self, sample_vec, knn=30, top_x=2):
        '''
        given a sample, give me the assigned class_id based on the voting routine
        '''
        predictions = self.annoy_index.get_nns_by_vector(sample_vec, knn, include_distances=False)
        return np.array(Counter(self.labels[predictions]).most_common())[:top_x,0]
    
    def assign_filename(self, sample_vec, knn=30, top_x=1):
        predictions = self.annoy_index.get_nns_by_vector(sample_vec, knn, include_distances=False)
        return np.array(Counter(self.filenames[predictions]).most_common())[:top_x,0]
    
    def _compute_confidence(self):
        '''
        given a data sample and its neighbors, along with the embeddings, computes 
        a certain confidence score.
        '''
        pass
    
    def _plot_tsne_lite(self, size):
        
        idx = np.random.choice(np.arange(len(self.labels)), size, replace=False)
        
        self.base_tsne = self.tsne_output[idx]
        self.labels_tsne = self.labels[idx]  
        
        plt.figure(figsize=(20,8))
        plt.scatter(self.base_tsne[:,0], self.base_tsne[:,1], c=self.base_tsne, alpha=0.2)
        plt.show()
    
    def plot_tsne(self, size):
        
        self._prepare_tsne()
        
        idx = np.random.choice(np.arange(len(self.labels)), size, replace=False)
        
        self.base_tsne = self.tsne_output[idx]
        self.labels_tsne = self.labels[idx]       
        
        plt.figure(figsize=(20,8))
        plt.scatter(self.base_tsne[:,0], self.base_tsne[:,1], c=self.labels_tsne, alpha=0.8)
        plt.show()

        
    
    def _load_population(self):
        if self.experimental_size > -1:
            self.base_population = np.load(self.base_population_path, allow_pickle=True).squeeze()[:self.experimental_size]
            self.labels = np.load(self.label_path, allow_pickle=True)[:self.experimental_size]
            self.filenames = np.load(self.filename_path, allow_pickle=True)[:self.experimental_size]
        else:
            self.base_population = np.load(self.base_population_path, allow_pickle=True).squeeze()
            self.labels = np.load(self.label_path, allow_pickle=True)
            self.filenames = np.load(self.filename_path, allow_pickle=True)
        