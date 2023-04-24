import tensorflow as tf

class Encoder2(tf.keras.Model):
    def __init__(self, view_ind, **kwargs):
        super(Encoder2, self).__init__(name=f'Encoder2_view_{view_ind}', **kwargs)
        self.view_index = view_ind

        self.dense_layers = [
            tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(1e-1)),
            tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(1e-1)),
            tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(1e-1)),
            tf.keras.layers.Dense(units=10, activation=None)
        ]

    def call(self, inputs):
        x = inputs #(,784)
        for layer in self.dense_layers:
            x = layer(x)

        return x #(,10)

    def model(self):
        x = tf.keras.layers.Input(shape=(784))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 
        
        
class TwoEncoders2(tf.keras.Model):
    def __init__(
        self,
        name="TwoViewsEncoder2",
        **kwargs
    ):
        super(TwoEncoders2, self).__init__(name=name, **kwargs)

        # Encoder
        self.encoder_v0 = Encoder2(view_ind=0)
        self.encoder_v1 = Encoder2(view_ind=1)

    def call(self, inputs):
        # The input 'inputs' is expected to be a dictionary with the keys used below
        inp_view_0 = inputs['nn_input_0'] #(M,784)
        inp_view_1 = inputs['nn_input_1']
        
        # Compute latent variables by feeding the data into the encoders
        latent_view_0 = self.encoder_v0(inp_view_0)
        latent_view_1 = self.encoder_v1(inp_view_1)
        
        # We just return the results in a dict. The loss is computed in the training script
        return {
            'latent_view_0':latent_view_0,
            'latent_view_1':latent_view_1,
        }
