import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model


class Encoder(object):
    def __init__(self, model_arc:str = 'vgg', inshape:tuple = (256,256,3), is_trainable:bool = False) -> None:
        self.model_arc = model_arc
        self.inshape = inshape
        self.is_trainable = is_trainable
        self.__supported_model_arc = ['vgg']

        self.vgg_layer_name = [
            'block1_conv2', 
            'block2_conv2', 
            'block3_conv2', 
            'block4_conv2',
        ]

    def get_encoder(self)->Model:
        if self.model_arc == 'vgg':
            self.encoder = VGG19(include_top=False, weights='imagenet', input_shape=self.inshape)
            self.encoder.trainable = self.is_trainable
            output = [tf.nn.relu(self.encoder.get_layer(layer).output) for layer in self.vgg_layer_name]
            return tf.keras.Model(inputs=self.encoder.input, outputs=output, name='vgg_encoder')
        else:
            raise ValueError(
                f'The model architecture {self.model_arc} is not supported. '
                f'Please choose one of the following: {self.__supported_model_arc}.'
            )

if __name__ == "__main__":
    encoder = Encoder(model_arc='vgg', inshape=(256,256,3), is_trainable=False)
    encoder_model = encoder.get_encoder()

    dummy_content_img = tf.random.uniform(shape=(1, 256, 256, 3))

    encoder_model.summary()
    content_feat     = encoder_model(dummy_content_img)
    for feat in content_feat:
        tf.print(feat.shape)

