import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, x:tf.Tensor)->tf.Tensor:
        assert len(x.shape) == 4, f'Input shape must be [batch, height, width, channels] but is {x.shape}'
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
    
    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({
            'padding': self.padding
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decoder(Model):
    '''
    Decoder model to decode the encoded style+content image passed through the AdaIN layer.
    TODO: Add different architectures.
    '''
    def __init__(self, name='StyleDecoder', *args, **kwargs):
        super(Decoder, self).__init__(name=name, *args, **kwargs)

        self.block4_conv1 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', name='block4_conv1')
        # self.block4_conv2 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', name='block4_conv2')
        # self.block4_conv3 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', name='block4_conv3')
        
        self.block3_conv1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', name='block3_conv1')
        self.block3_conv2 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', name='block3_conv2')
        self.block3_conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', name='block3_conv3')

        self.block2_conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', name='block2_conv1')
        self.block2_conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', name='block2_conv2')

        self.block1_conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', name='block1_conv1')
        self.block1_conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', name='block1_conv2')

        # Reflection padding
        self.reflection_padding = ReflectionPadding2D(padding=(1, 1))

        self.out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', name='out')
    
    def call(self, inputs:tf.Tensor, training=None):
        assert inputs.shape.rank == 4, f'The shape of the input must be (batch_size, height, width, channels).'

        x = self.block4_conv1(inputs)
        x = self.reflection_padding(x)
        # x = self.block4_conv2(x)
        # x = self.block4_conv3(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        
        x = self.block3_conv1(x)
        x = self.reflection_padding(x)
        x = self.block3_conv2(x)
        x = self.reflection_padding(x)
        x = self.block3_conv3(x)
        x = self.reflection_padding(x)
        x = layers.UpSampling2D(size=(2, 2))(x)

        x = self.block2_conv1(x)
        x = self.reflection_padding(x)
        x = self.block2_conv2(x)
        x = self.reflection_padding(x) 
        x = layers.UpSampling2D(size=(2, 2))(x)

        x = self.block1_conv1(x)
        x = self.reflection_padding(x)
        x = self.block1_conv2(x)
        x = self.reflection_padding(x)

        x = self.out(x)

        return x

    def summary(self):
        inputs = tf.keras.Input(shape=(32, 32, 512))
        model = Model(inputs=[inputs], outputs=self(inputs), name=self.name)
        model.summary()
    
    def get_config(self):
        return super(Decoder, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

if __name__ == '__main__':
    dummy_content_feat = tf.random.normal(shape=(1, 32, 32, 512))
    decoder = Decoder()

    decoder.summary()
    recon_img = decoder(dummy_content_feat)
    print(recon_img.shape)





