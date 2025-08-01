#customImputerLayerDefinition.py
import tensorflow as tf

class ImputerLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImputerLayer, self).__init__(**kwargs)
        self.min_values = None

    def adapt(self, data):
        # Calculate min values across non-NaN entries
        self.min_values = tf.reduce_min(tf.where(tf.math.is_nan(data), tf.float32.max, data), axis=0)

    def call(self, inputs):
        # Replace NaN values with minimum values
        return tf.where(tf.math.is_nan(inputs), self.min_values, inputs)