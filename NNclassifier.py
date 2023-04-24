import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax,Conv2D, MaxPool2D, Flatten    

class Fullconnection(tf.keras.Model):
    def __init__(self,config, **kwargs):
        super(Fullconnection, self).__init__(name=f'Fullconnection', **kwargs)
        self.config = config

        self.cnn_layers = [
            Dense(10,None),
            Softmax()
        ]

        
    def call(self, inputs):
        x = inputs
        for layer in self.cnn_layers :
            x = layer(x)
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.config))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 

class CNNcla(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CNNcla, self).__init__(name=f'CNNcla', **kwargs)

        self.cnn_layers = [
            Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding="same"),
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),
            Conv2D(filters=4, kernel_size=(5,5), activation='relu', padding="same"),
            MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),
            Flatten(),
            Dense(units=10,activation = None),
        ]

        
    def call(self, inputs):
        x = inputs
        for layer in self.cnn_layers :
            x = layer(x)
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(28,28,1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 

class NNcla(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(NNcla, self).__init__(name=f'NNcla', **kwargs)
        self.config = config

        self.dense_layers = [
            tf.keras.layers.Dense(
                dim,
                activation=activ,
            ) for (dim, activ) in self.config
        ]

    def call(self, inputs):
        x = inputs #(,784)
        for layer in self.dense_layers:
            x = layer(x)

        return x #(,10)

    def model(self):
        x = tf.keras.layers.Input(shape=(784)) #it depends on the size of outputs of model
        return tf.keras.Model(inputs=[x], outputs=self.call(x))