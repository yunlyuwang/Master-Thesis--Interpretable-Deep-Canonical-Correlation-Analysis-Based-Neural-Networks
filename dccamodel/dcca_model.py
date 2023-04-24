import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Encoder, self).__init__(name=f'Encoder_view_{view_ind}', **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = [
            tf.keras.layers.Dense(
                dim,
                kernel_regularizer=tf.keras.regularizers.L2(1e-10),  #L1 or L2, avoid the overfiting, but the result doesn't change lot.
                activation=activ,
            ) for (dim, activ) in self.config
        ]

    def call(self, inputs):
        x = inputs #(,784)
        for layer in self.dense_layers:
            x = layer(x)

        return x #(,40)

    def model(self):
        x = tf.keras.layers.Input(shape=(784)) 
        return tf.keras.Model(inputs=[x], outputs=self.call(x)) 
        
        
class TwoEncoders(tf.keras.Model):
    def __init__(
        self,
        encoder_config,
        name="TwoViewsEncoder",
        **kwargs
    ):
        super(TwoEncoders, self).__init__(name=name, **kwargs)
        self.encoder_config = encoder_config

        # Encoder
        self.encoder_v0 = Encoder(encoder_config, view_ind=0)
        self.encoder_v1 = Encoder(encoder_config, view_ind=1)

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



class Decoder(tf.keras.Model):
    def __init__(
        self,
        decoder_config,
        name="Decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_config = decoder_config

        # decoder
        self.decoder_v0 = Encoder(decoder_config, view_ind=0)
        self.decoder_v1 = Encoder(decoder_config, view_ind=1)

    def call(self, outputs):
        # The input 'inputs' is expected to be a dictionary with the keys used below
        outp_view_0 = outputs['latent_view_0']
        outp_view_1 = outputs['latent_view_1']
        
        # Compute latent variables by feeding the data into the encoders
        latent_out_view_0 = self.decoder_v0(outp_view_0)
        latent_out_view_1 = self.decoder_v1(outp_view_1)
        
        # We just return the results in a dict. The loss is computed in the training script
        return {
            'latent_out_view_0':latent_out_view_0,
            'latent_out_view_1':latent_out_view_1,
        }
