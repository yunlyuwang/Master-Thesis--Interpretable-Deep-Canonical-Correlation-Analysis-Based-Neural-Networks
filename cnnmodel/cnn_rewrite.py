import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

class DeepCCAcnn(tf.keras.Model):
    def __init__(
        self,
        name="DeepCCAcnn",
        **kwargs
    ):
        super(DeepCCAcnn, self).__init__(name=name, **kwargs)

        self.cnn_layers = [
            Conv2D(filters=8, kernel_size=(3,3), activation=None, padding="same"),  #(,28,28,8)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,14,14,8)
            Conv2D(filters=6, kernel_size=(3,3), activation=None, padding="same"),  #(,14,14,6)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,7,7,6)
            Conv2D(filters=1, kernel_size=(3,3), activation=None, padding="same"),  #(,7,7,1)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,3,3,1)
            Flatten()                                                               #(,9)
        ]

    def call(self, inputs):
        inp_view_0 = inputs['nn_input_0'] #(50,784)
        inp_view_1 = inputs['nn_input_1']

        M = inp_view_0.shape[0] #50
        x = tf.cast(tf.reshape(inp_view_0,[M, 28, 28, 1]), dtype=tf.float32)          
        y = tf.cast(tf.reshape(inp_view_1,[M, 28, 28, 1]), dtype=tf.float32)   

        for layer in self.cnn_layers :
            x = layer(x)
            y = layer(y)

        return {
            'latent_view_0':x,
            'latent_view_1':y,
        }