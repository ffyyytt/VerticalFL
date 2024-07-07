import tensorflow as tf

class TriggerLayer(tf.keras.layers.Layer):
    def __init__(self, windowSize, **kwargs):
        super().__init__(**kwargs)
        self.windowSize = windowSize

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.windowSize, self.windowSize), initializer='glorot_uniform', trainable=True)

    def call(self, inputs, training):
        images, position = inputs
        for k in range(len(images)):
            for i in range(self.windowSize):
                for j in range(self.windowSize):
                    images[k, position[k][0]+i, position[k][1]+j] = self.W[i, j]
        return images