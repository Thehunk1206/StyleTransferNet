import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


def preprocess_input(x: tf.Tensor, swap_channel:bool=True) -> tf.Tensor:
    '''
    Preprocess the input image.
    '''
    assert x.shape[-1] == 3, f'The shape of x must be (batch_size, height, width, channel). But got {x.shape}.'

    
    # Swap the channel of the image
    if swap_channel:
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant([103.939, 116.779, 123.68])
        return x
    else:
        x = x - tf.constant([123.68, 116.779, 103.939])
        return x

def process_output(x:tf.Tensor, swap_channel:bool = True):
    '''
    Process output 
    '''
    if swap_channel:
        x = x + tf.constant([103.939, 116.779, 123.68])
        x = tf.reverse(x, axis=[-1])
        return x
    else:
        x = x + tf.constant([123.68, 116.779, 103.939])
        return x