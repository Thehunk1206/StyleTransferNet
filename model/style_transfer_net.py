import os
from typing import Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

from model.encoder import Encoder
from model.decoder import Decoder
from model.adaIn import AdaIn

class StyleTransferNet(Model):
    def __init__(self, IMG_H:int=256, IMG_W:int=256, encoder_model_arc:str = 'vgg', name:str='StyleTransferNet', *args, **kwargs):
        super(StyleTransferNet, self).__init__(name=name, *args, **kwargs)
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.encoder_model_arc = encoder_model_arc
        self.optimizer = None
        self.content_loss = None
        self.style_loss = None
        self.loss_weight = 1.0

        # Define StyleTransfer network
        self.encoder = Encoder(model_arc=self.encoder_model_arc, inshape=(self.IMG_H, self.IMG_W, 3))
        self.encoder_model = self.encoder.get_encoder()

        self.adaIn = AdaIn()

        self.decoder = Decoder()

    def call(self, content_img:tf.Tensor, style_img:tf.Tensor, training=None)->Tuple[tf.Tensor, Tuple[tf.Tensor,...], tf.Tensor, Tuple[tf.Tensor,...], Tuple[tf.Tensor,...]]:
        content_feat = self.encoder_model(content_img) # Extract content features from content image
        style_feat = self.encoder_model(style_img) # Extract style features from style image

        adain_vec = self.adaIn(content_feat[-1], style_feat[-1]) # Normalize content features, shift and scale with style features(AdaIn)

        stylized_img = self.decoder(adain_vec, training=training) # Generate stylized image from normalized content features

        stylized_feat = self.encoder_model(stylized_img) # Extract stylized features from stylized image

        return stylized_img, stylized_feat, adain_vec, style_feat, content_feat
    
    def compile(
            self, 
            optimizer=Optimizer, 
            content_loss=Loss, 
            style_loss=Loss,
            loss_weight:float=2.0, 
            **kwargs
        ):
        super(StyleTransferNet, self).compile(**kwargs)
        self.optimizer = optimizer
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.loss_weight = loss_weight

    @tf.function
    def train_step(self, content_img:tf.Tensor, style_img:tf.Tensor)->tuple:
        '''
        Forward pass, calculates total loss, and calculate gradients with respect to loss.
        args:
            content_img: tf.Tensor, shape=(batch_size, IMG_H, IMG_W, 3)
            style_img: tf.Tensor, shape=(batch_size, IMG_H, IMG_W, 3)  
        returns:
            total_loss, content_loss, style_loss
        '''
        assert self.optimizer is not None, 'optimizer is not defined, call compile() first'
        assert self.content_loss is not None, 'content_loss is not defined, call compile() first'
        assert self.style_loss is not None, 'style_loss is not defined, call compile() first'

        with tf.GradientTape() as tape:
            _, stylized_feat, adain_vec, style_feat, _ = self.call(content_img, style_img, training=True)
            content_loss = self.content_loss(stylized_feat[-1], adain_vec)
            style_loss = self.style_loss(stylized_feat, style_feat)
            total_loss = content_loss + self.loss_weight * style_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return total_loss, content_loss, style_loss

    def summary(self):
        inp1 = tf.keras.Input(shape=(self.IMG_H, self.IMG_W, 3))
        inp2 = tf.keras.Input(shape=(self.IMG_H, self.IMG_W, 3))

        style_transfer_net = Model(inputs=[inp1, inp2], outputs=self.call(inp1, inp2), name='style_transfer_net')
        style_transfer_net.summary()
        self.decoder.summary()
        self.encoder_model.summary()
        
    def get_config(self):
        return {
            'IMG_H': self.IMG_H,
            'IMG_W': self.IMG_W,
            'encoder_model_arc': self.encoder_model_arc
        }

    @classmethod
    def from_config(cls, config:dict):
        return cls(**config)
    

if __name__ == "__main__":
    dummy_content_img = tf.random.uniform(shape=(1, 256, 256, 3))
    dummy_style_img = tf.random.uniform(shape=(1, 256, 256, 3))

    style_transfer_net = StyleTransferNet(IMG_H=256, IMG_W=256, encoder_model_arc='vgg')
    style_transfer_net.summary()

    stylized_img, stylized_feat, adain_vec, style_feat, content_feat = style_transfer_net(dummy_content_img, dummy_style_img)

    tf.print(
        f'\nstylized_img.shape: {stylized_img.shape}\n',
        f'stylized_feat shape:  {[f.shape for f in stylized_feat]}\n',
        f'adain_vec.shape:      {adain_vec.shape}\n',
        f'style_feat.shape:     {[f.shape for f in style_feat]}\n',
        f'content_feat.shape:   {[f.shape for f in content_feat]}\n',
    )
    