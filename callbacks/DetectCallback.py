import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import pairwise_distances

class DetectCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, labels, nparty, p=0.8):
        self.data = data
        self.labels = labels
        self.nparty = nparty
        self.p = p
        self.isAttacked = False

    def on_epoch_end(self, epoch, logs={}):
        model = tf.keras.models.Model(inputs = self.model.inputs, 
                                      outputs = [self.model.get_layer(f'feature_{i}').output for i in range(self.nparty)])
        
        yPred = model.predict(self.data, verbose = False)

        for client in range(self.nparty):
            features = yPred[client]
            distances = pairwise_distances(features)
            distances += np.eye(len(features))*np.max(distances)
            nearestNeighbor = np.argmax(distances, axis=1)

            for label in range(len(set(self.labels))):
                idx = np.where(self.labels==label)
                neighbors = self.labels[nearestNeighbor[idx]]
                if np.mean(neighbors == label) < self.p:
                    print(f"Detected client {client}", np.mean(neighbors == label))
                    self.isAttacked = True
                    self.model.stop_training = True

        self.goodModel = tf.keras.models.clone_model(self.model)