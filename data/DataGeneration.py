import cv2
import random
import numpy as np
import tensorflow as tf

class BaseDataGeneration(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batchsize, shuffle = False, n_party = 2, **kwargs):
        self.ids = list(range(images.shape[0]))
        self.images = tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(images, tf.float32), mode="tf").numpy()
        self.labels = labels
        self.batchsize = batchsize
        
        self.shuffle = shuffle
        self.n_party = n_party

        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.images) // self.batchsize + int(len(self.images) % self.batchsize != 0)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.ids = sorted(self.ids, key=lambda k: random.random())
            
    def __getitem__(self, index):
        idx = self.ids[index*self.batchsize: min((index+1)*self.batchsize, len(self.ids))]
        images = self.images[idx]
        labels = self.labels[idx]

        X = {}
        for i in range(self.n_party):
            if i != self.n_party-1:
                X[f"image_{i}"] = images[:, :, i*round(images.shape[2]/self.n_party):(i+1)*round(images.shape[2]/self.n_party)]
            else:
                X[f"image_{i}"] = images[:, :, i*round(images.shape[2]/self.n_party):]
        return X, {"output": labels}