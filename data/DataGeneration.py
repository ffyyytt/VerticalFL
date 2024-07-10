import cv2
import random
import numpy as np
import tensorflow as tf
from model.model import *

class BaseDataGeneration(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batchsize, shuffle = False, n_party = 2, **kwargs):
        self.ids = list(range(images.shape[0]))
        self.images = images
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


class FindTriggerDataGeneration(tf.keras.utils.Sequence):
    def __init__(self, x_sub_s, positions, x_t, windowSize, batchsize, party, n_party, **kwargs):
        self.x_sub_s = x_sub_s
        self.positions = positions
        self.x_t = x_t
        self.windowSize = windowSize
        self.batchsize = batchsize
        self.party = party
        self.n_party = n_party
        self.ids = random.choices(list(range(len(self.x_sub_s))), k=len(self.x_t))
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.x_t) // self.batchsize + int(len(self.x_t) % self.batchsize != 0)
    
    def on_epoch_end(self):
        self.ids = random.choices(list(range(len(self.x_sub_s))), k=len(self.x_t))

    def position_to_masks(self, position, imageShape):
        masks = np.zeros([self.windowSize, self.windowSize, 3, imageShape[0], imageShape[1], 3], dtype="float32")
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                for k in range(3):
                    masks[i, j, k][position[0]+i, position[1]+j, k] = 1
        return masks
            
    def __getitem__(self, index):
        idx = self.ids[index*self.batchsize: min((index+1)*self.batchsize, len(self.ids))]
        x_sub_s = self.x_sub_s[idx][:, :, self.party*round(self.x_sub_s.shape[2]/self.n_party):(self.party+1)*round(self.x_sub_s.shape[2]/self.n_party)]
        x_t = self.x_t[idx][:, :, self.party*round(self.x_sub_s.shape[2]/self.n_party):(self.party+1)*round(self.x_sub_s.shape[2]/self.n_party)]
        masks = np.array([self.position_to_masks(self.positions[i], x_sub_s.shape[1:3]) for i in idx])
        return {"x_sub_s": x_sub_s, "x_t": x_t, "masks": masks}, {"output": np.array([0.0]*len(idx))}
    
def findTrigger(model, p, images, labels, positions, targetClass, sourceClass, windowSize, batch, nparty, partyIdx, strategy, lr, momentum, epochs):
    x_sub_s_idx = np.where(np.argmax(labels, axis=1) == sourceClass)[0]
    x_sub_s_idx = random.choices(x_sub_s_idx, k = round(p*len(x_sub_s_idx)))
    x_t_idx = np.where(np.argmax(labels, axis=1) == targetClass)[0]
    x_sub_s = images[x_sub_s_idx]
    x_t = images[x_t_idx]
    x_sub_positions = positions[x_sub_s_idx]

    triggerData = FindTriggerDataGeneration(x_sub_s, x_sub_positions, x_t, windowSize, batch, partyIdx, nparty)

    with strategy.scope():
        extractor = buildExtractModel(model, partyIdx)
    targetFeatures = extractor.predict(triggerData, verbose = 0)

    with strategy.scope():
        triggerModel = buildTriggerModel(model, windowSize, targetFeatures, partyIdx)
        triggerModel.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
                    loss = {'output': tf.keras.losses.MeanSquaredError()},
                    metrics = {"output": [tf.keras.metrics.MeanAbsoluteError()]})
    
    triggerModel.fit(triggerData, epochs=epochs, verbose = 0)
    return triggerModel.layers[2].W.numpy()

class AttackDataGeneration(tf.keras.utils.Sequence):
    def __init__(self, model, p, images, labels, positionsDict, targetClass, sourceClass, windowSize, batchsize, strategy, lr, momentum, epochs, shuffle = False, n_party = 2, **kwargs):
        self.model = model
        self.p = p
        self.ids = list(range(images.shape[0]))
        self.images = images
        self.labels = labels
        self.batchsize = batchsize
        
        self.shuffle = shuffle
        self.n_party = n_party

        self.positionsDict = positionsDict
        self.targetClass = targetClass
        self.sourceClass = sourceClass
        self.windowSize = windowSize

        self.strategy = strategy
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

        self.on_epoch_end()
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.images) // self.batchsize + int(len(self.images) % self.batchsize != 0)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.ids = sorted(self.ids, key=lambda k: random.random())

        self.triggers = {k: findTrigger(self.model, self.p, self.images, self.labels, self.positionsDict[k], 
                                     self.targetClass, self.sourceClass, self.windowSize, self.batchsize, 
                                     self.n_party, k, self.strategy, self.lr, self.momentum, self.epochs) for k in self.positionsDict.keys()}
        
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

            if i in self.positionsDict:
                for j in range(len(idx)):
                    if np.argmax(labels, axis=1)[j] == self.sourceClass:
                        position = self.positionsDict[i][idx[j]]
                        X[f"image_{i}"][j][position[0]:position[0]+self.windowSize, position[1]:position[1]+self.windowSize] += self.triggers[i]
        return X, {"output": labels}
    
class ASRDataGeneration(tf.keras.utils.Sequence):
    def __init__(self, images, labels, positionsDict, triggers, batchsize, n_party = 2, **kwargs):
        self.ids = list(range(images.shape[0]))
        self.images = images
        self.labels = labels
        self.positionsDict = positionsDict
        self.triggers = triggers
        self.batchsize = batchsize
        
        self.n_party = n_party

        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.images) // self.batchsize + int(len(self.images) % self.batchsize != 0)
            
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

            if i in self.positionsDict:
                for j in range(len(idx)):
                    position = self.positionsDict[i][idx[j]]
                    X[f"image_{i}"][j][position[0]:position[0]+self.windowSize, position[1]:position[1]+self.windowSize] += self.triggers[i]
        return X, {"output": labels}