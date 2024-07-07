import tensorflow as tf

class TriggerLayer(tf.keras.layers.Layer):
    def __init__(self, windowSize, **kwargs):
        super().__init__(**kwargs)
        self.windowSize = windowSize

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.windowSize, self.windowSize, 3), initializer='glorot_uniform', trainable=True)

    def call(self, inputs, training):
        images, masks = inputs
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                for k in range(3):
                    images += self.W[i, j, k]*masks[i, j, k]
        return images