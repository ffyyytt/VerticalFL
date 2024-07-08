import tensorflow as tf

class TriggerLayer(tf.keras.layers.Layer):
    def __init__(self, windowSize, **kwargs):
        super().__init__(**kwargs)
        self.windowSize = windowSize

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.windowSize, self.windowSize, 3), initializer='glorot_uniform', trainable=True)
        self.W = tf.clip_by_value(tf.math.abs(self.W)*255, 0, 255)

    def call(self, inputs, training):
        W = tf.clip_by_value(self.W, 0, 255)
        images, masks = inputs
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                for k in range(3):
                    images += W[i, j, k]*masks[i, j, k]
        return images