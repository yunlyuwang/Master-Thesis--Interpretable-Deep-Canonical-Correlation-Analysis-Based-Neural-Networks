import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

class CnnEncoder(tf.keras.Model):
    def __init__(self, view_ind, **kwargs):
        super(CnnEncoder, self).__init__(name=f'CnnEncoder_view_{view_ind}', **kwargs)
        self.view_index = view_ind

        self.cnn_layers = [
            Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(1e-6)), #(,28,28,8)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,14,14,8)
            Conv2D(filters=6, kernel_size=(5,5), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(1e-6)), #(,14、,14,6)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,7,7,6)
            Conv2D(filters=4, kernel_size=(5,5), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(1e-6)),  #(,7,7,4)
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),             #(,3,3,4)
            Conv2D(filters=2, kernel_size=(5,5), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(1e-6)),  #(,3,3,2)
            Flatten(),                                                         #(,18)                  
            ]

        
    def call(self, inputs):
        x = inputs
        for layer in self.cnn_layers :
            x = layer(x)
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(28,28,1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 
                                                                                              

class TwoCnnEncoders(tf.keras.Model):
    def __init__(
        self,
        name="TwoViewsCnnEncoder",
        **kwargs
    ):
        super(TwoCnnEncoders, self).__init__(name=name, **kwargs)

        # Encoder
        self.cnnencoder_v0 = CnnEncoder(view_ind=0)
        self.cnnencoder_v1 = CnnEncoder(view_ind=1)

    def call(self, inputs):
        # The input 'inputs' is expected to be a dictionary with the keys used below
        inp_view_0 = inputs['nn_input_0']
        inp_view_1 = inputs['nn_input_1']

        M = inp_view_0.shape[0] #50
        #N = inp_view_0.shape[1] #784 inputs=(50,784)
        
        inp_view_0 = tf.cast(tf.reshape(inp_view_0,[M, 28, 28, 1]), dtype=tf.float32)  
        inp_view_1 = tf.cast(tf.reshape(inp_view_1,[M, 28, 28, 1]), dtype=tf.float32)
          
        # Compute latent variables by feeding the data into the encoders
        latent_view_0 = self.cnnencoder_v0(inp_view_0)
        latent_view_1 = self.cnnencoder_v1(inp_view_1)
        
        # We just return the results in a dict. The loss is computed in the training script
        return {
            'latent_view_0':latent_view_0,
            'latent_view_1':latent_view_1,
        }