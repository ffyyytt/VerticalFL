import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import pairwise_distances

class DetectCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, labels, nparty):
        self.data = data
        self.labels = labels
        self.nparty = nparty

    def on_epoch_end(self, epoch, logs={}):
        model = tf.keras.models.Model(inputs = self.model.inputs, 
                                      outputs = [self.model.get_layer(f'feature_{i}').output for i in range(self.nparty)])
        
        yPred = model.predict(self.data)

        for i in range(self.nparty):
            features = yPred[i]
            distances = pairwise_distances(features)
            distances += np.eye(len(features))*np.max(distances)
            nearestNeighbor = np.argmax(distances, axis=1)
            print(Counter([str([self.labels[i], self.labels[nearestNeighbor[i]]]) for i in range(len(features))]))