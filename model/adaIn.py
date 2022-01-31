import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Layer

'''
A tf.keras Layer to implement AdaIN(Adaptive Instance Normalization).
'''

class AdaIn(Layer):
    def __init__(self,name='AdaIn', **kwargs):
        super(AdaIn, self).__init__(name=name, **kwargs)
        self.epsilon = 1e-5
    
    def call(self, content_vec:tf.Tensor, style_vec:tf.Tensor, *args, **kwargs):
        '''
        The forward pass of AdaIn.
        args:
            content_vec: tf.Tensor, the content image
            style_vec: tf.Tensor, the style image
        '''
        assert len(content_vec.shape)   == 4, f'The shape of content_vec must be (batch_size, height, width, channel). But got {content_vec.shape}.'
        assert len(style_vec.shape)     == 4, f'The shape of style_vec must be (batch_size, height, width, channel). But got {style_vec.shape}.'

        # Calculate the mean and variance of the content image
        content_mean, content_var  = tf.nn.moments(content_vec, axes=[1,2], keepdims=True)
        # Calculate the mean and variance of the style image
        style_mean, style_var = tf.nn.moments(style_vec, axes=[1,2], keepdims=True)

        content_std = tf.math.sqrt(content_var + self.epsilon)
        style_std   = tf.math.sqrt(style_var + self.epsilon)

        # Calculate the normalized content image
        normalized_content_vec = (content_vec - content_mean) / content_std

        return normalized_content_vec * style_std + style_mean
    
    def get_config(self):
        return super(AdaIn,self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    from tensorflow.keras.applications import VGG19, VGG16
    # Test AdaIn
    dummy_content_img   = tf.random.uniform(shape=(1, 224, 224, 3))
    dummy_style_img     = tf.random.uniform(shape=(1, 224, 224, 3))

    vgg16 = VGG16(include_top=False, input_shape=(224,224,3))
    vgg16.trainable = False
    adain = AdaIn()

    vgg16.summary()

    content_vec = vgg16(dummy_content_img)
    style_vec   = vgg16(dummy_style_img)

    adain_output = adain.call(content_vec, style_vec)
    print(adain_output.shape)

