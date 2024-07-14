import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import pairwise_distances

class DetectCallback(tf.keras.callbacks.Callback):
    """A simple backdoor attack detector."""
    """Since target features and source features are similar (by triggers). So we detect if the features are similar or not."""
    """This step can also be done during the training process, but it would have a high time complexity. And I don't have any callbacks in this project yet so I wanted to challenge myself by writing a callback."""
    def __init__(self, data, labels, nparty, p=0.8):
        self.data = data
        self.labels = labels
        self.nparty = nparty
        self.p = p
        self.isAttacked = False

    def on_epoch_end(self, epoch, logs={}):
        # remove classfier layer
        model = tf.keras.models.Model(inputs = self.model.inputs, 
                                      outputs = [self.model.get_layer(f'feature_{i}').output for i in range(self.nparty)])
        
        yPred = model.predict(self.data, verbose = False)

        for client in range(self.nparty):
            features = yPred[client]

            # Find the nearest neighbor of each sample
            distances = pairwise_distances(features)
            distances += np.eye(len(features))*np.max(distances) # not itself
            nearestNeighbor = np.argmax(distances, axis=1)
            
            """For each class, get all neighbors of each sample belong to that class. 
            Check the label of all neighbors, If other label accounts for a significant amount -> return malicious
            """
            for label in range(len(set(self.labels))):
                idx = np.where(self.labels==label)
                neighbors = self.labels[nearestNeighbor[idx]]
                if np.mean(neighbors == label) < self.p:
                    print(f"Detected client {client}", np.mean(neighbors == label))
                    self.isAttacked = True
                    self.model.stop_training = True

        self.goodModel = tf.keras.models.clone_model(self.model)