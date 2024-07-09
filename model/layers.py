import tensorflow as tf

class TriggerLayer(tf.keras.layers.Layer):
    def __init__(self, windowSize, **kwargs):
        super().__init__(**kwargs)
        self.windowSize = windowSize

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.windowSize, self.windowSize, 3), initializer='glorot_uniform', trainable=True)
        self.W = tf.clip_by_value(tf.math.abs(self.W)*255, 0, 255)

    def call(self, inputs, training=False):
        W = tf.clip_by_value(self.W, 0, 255)
        images, masks = inputs
        for i in range(self.windowSize):
            for j in range(self.windowSize):
                for k in range(3):
                    images += W[i, j, k]*masks[i, j, k]
        return images
    
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, features, norm=False, **kwargs):
        super().__init__(**kwargs)
        self.features = tf.convert_to_tensor(features)
        self.norm = norm
    
    def call(self, inputs, training=False):
        f1, f2 = inputs, self.features
        if self.norm:
            f1 = tf.nn.l2_normalize(f1, axis=1)
            f2 = tf.nn.l2_normalize(f2, axis=1)
        f1 = tf.tile(tf.expand_dims(f1, 2), [1, 1, f2.shape[0]])
        f2 = tf.transpose(f2)
        return tf.reduce_mean(tf.reduce_sum( tf.math.pow( f1 - f2, 2 ), axis=1), axis=1)