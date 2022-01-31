import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.losses import Loss

'''
Class to create a custom loss fucntion by subclassing Loss class from tf.keras.losses module.
'''

class ContentLoss(Loss):
    def __init__(self, name='ContentLoss', **kwargs):
        super(ContentLoss, self).__init__(name=name, **kwargs)

    def call(self, stylized_feat_vec:tf.Tensor, adain_vec:tf.Tensor)-> tf.Tensor:
        '''
        Calculate the content loss as given in the paper "Arbitrary Style Transfer with AdaIn".
        The content loss is calculated as the euclidean distance between stylized_vec and adain_vec vectors.
        args:
            stylized_feat_vec: tf.Tensor, the output of the Encoder network 
            after passing stylized image through the pre-trained Encoder network

            adain_vec: tf.Tensor, the output of the AdaIn layer
        return:
            content_loss: tf.Tensor, the content loss
        '''
        assert len(stylized_feat_vec.shape) == 4, f'Dims of stylized_feat_vec must be 4. But got {stylized_feat_vec.shape.rank}.'
        assert len(adain_vec.shape) == 4, f'Dims of adain_vec must be 4. But got {adain_vec.shape.rank}.'

        # Calculate the content loss
        content_loss = tf.math.sqrt(tf.reduce_sum(tf.math.square(stylized_feat_vec - adain_vec)))

        return content_loss
    
    def get_config(self):
        return super(ContentLoss,self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class StyleLoss(Loss):
    def __init__(self, name='StyleLoss', **kwargs):
        super(StyleLoss, self).__init__(name=name, **kwargs)
    
    def __call__(self, stylized_feat_vecs:list, style_feat_vecs:list)-> tf.Tensor:
        '''

        '''
        assert len(stylized_feat_vecs) == len(style_feat_vecs), f'The number of stylized_feat_vecs and style_feat_vecs must be the same.'

        # Calculate the style loss
        style_loss = 0
        for stylized_feat_vec, style_feat_vec in zip(stylized_feat_vecs, style_feat_vecs):
            style_feat_m, style_feat_var        = tf.nn.moments(style_feat_vec, axes=[1,2])
            stylized_feat_m, stylized_feat_var  = tf.nn.moments(stylized_feat_vec, axes=[1,2])

            style_feat_std      = tf.math.sqrt(style_feat_var)
            stylized_feat_std   = tf.math.sqrt(stylized_feat_var)

            style_loss += tf.norm(stylized_feat_m - style_feat_m) + tf.norm(stylized_feat_std - style_feat_std) 
        
        return style_loss
    
    def get_config(self):
        return super(StyleLoss,self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    from tensorflow.keras import Model
    from tensorflow.keras.applications import VGG19, VGG16

    vgg_layers = [
        'block1_conv2', 
        'block2_conv2', 
        'block3_conv2', 
        'block4_conv2',
        'block5_conv1'
    ]

    tf.random.set_seed(1)
    # Test Content loss
    dummy_content_img   = tf.random.uniform(shape=(4, 256, 256, 3))
    dummy_style_img     = tf.random.uniform(shape=(4, 256, 256, 3))
    dummy_stylized_img  = tf.random.uniform(shape=(4, 256, 256, 3))

    # Create VGG19 Feature Extractor model
    vgg = VGG19(include_top=False, input_shape=(256,256,3))
    vgg.trainable = False
    output = [vgg.get_layer(layer).output for layer in vgg_layers]
    vgg_model = Model(inputs=vgg.input, outputs=output)
    vgg_model.summary()

    style_feat       = vgg_model(dummy_style_img)
    stylized_feat    = vgg_model(dummy_stylized_img)
    content_feat     = vgg_model(dummy_content_img)

    content_loss = ContentLoss()
    style_loss = StyleLoss()

    loss_s = style_loss(stylized_feat, style_feat)
    loss_c = content_loss(stylized_feat[-1], content_feat[-1])

    tf.print(f'Style Loss: {loss_s}', f'Content Loss: {loss_c}')
