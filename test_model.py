import os
import argparse
from sympy import content

from typer import style
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import models


from utils.dataset import TfdataPipeline
from utils.preprocess_io import preprocess_input, process_output
from model.adaIn import AdaIn
from model.encoder import Encoder

from matplotlib import pyplot as plt

# A fucntion to plot stlye, content and stylized images
def plot_images(
    content_img:tf.Tensor,
    style_img:tf.Tensor,
    stylized_img:tf.Tensor,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))
    axes[0].imshow(content_img.numpy()[0])
    axes[0].set_title('Content Image')
    axes[1].imshow(style_img.numpy()[0])
    axes[1].set_title('Style Image')
    axes[2].imshow(stylized_img.numpy()[0])
    axes[2].set_title('Stylized Image')
    plt.show()

def read_image(path:str, img_h:int = 256, img_w:int=256) -> tf.Tensor:
        '''
        Load the content and style images.
        args:
            path: str, the path of the image
        return:
            image_tensor: tf.Tensor, the image
        '''
        
        # Read raw image from path
        image = tf.io.read_file(path)

        # Decode the raw image
        image = tf.io.decode_image(image, channels=3, expand_animations=False)

        # Change the dtype of the image to float32
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize the image to the desired size
        image = tf.image.resize(image, [img_h,img_w], method=tf.image.ResizeMethod.BICUBIC)
        
        # Normalize the image
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        # Expand the image to 4D
        image = tf.expand_dims(image, axis=0)

        return image

def get_model(model_path: str):
    assert os.path.exists(model_path), f'{model_path} does not exist'

    tf.print(
        "[info] loading model from disk...."
    )
    model = models.load_model(model_path)

    tf.print(
        "loaded model {}".format(model)
    )
    return model

def run_test(
    dataset_dir: str,
    model_path:str, 
    test_out_path:str='test_results/'
):  
    if not os.path.exists(test_out_path):
        os.mkdir(test_out_path)
    
    nst_model = get_model(model_path)

    encoder = Encoder()
    encoder_model = encoder.get_encoder()
    adain = AdaIn()


    # tf.print('[info] loading dataset...')
    # tf_dataset = TfdataPipeline(
    #     BASE_DATASET_DIR=dataset_dir,
    #     batch_size=1
    # )
    # test_ds = tf_dataset.data_loader(dataset_type='test', do_augment=False)


    tf.print('[info] testing...')

    style_img = read_image('style_images/abstract-painting.jpg')
    content_img = read_image('content_images/IMG_20210123_155649.jpg')

    style_img_p = preprocess_input(style_img*255.0)
    content_img_p = preprocess_input(content_img*255.0)

    output,_,_,_,_ = nst_model(content_img_p, style_img_p)
    tf.print(output)

    output = tf.clip_by_value(output, 0.0, 255.0)
    output = output/255.0
    # output = (output - tf.reduce_min(output)) / (tf.reduce_max(output) - tf.reduce_min(output))

    # for content_img, style_img in test_ds.take(1):

    #     content_img = tf.image.resize(content_img, (256, 256))
    #     style_img = tf.image.resize(style_img, (256, 256))

    #     content_img_p = preprocess_input(content_img*255.0)
    #     style_img_p = preprocess_input(style_img*255.0)

    #     output,_,_,_,_ = nst_model(content_img_p, style_img_p)

    #     # min max normalization
    #     # output = (output - tf.reduce_min(output)) / (tf.reduce_max(output) - tf.reduce_min(output))

    #     # Clipping the values between 0 and 255
    #     output = tf.clip_by_value(output, 0.0, 255.0)
    #     output = output/255.0

    #     tf.print(f'[info] output shape: {output.shape}')
    #     tf.print(f'[info] Max value: {tf.reduce_max(output)}')
    #     tf.print(f'[info] Min value: {tf.reduce_min(output)}')
    #     tf.print(f'[info] Mean value: {tf.reduce_mean(output)}')

    plot_images(content_img, style_img, output)

if __name__ == "__main__":
    model_path = 'trained_model/StyleTransferNet_iteration_4'

    dataset_dir = 'style_content_dataset/'

    run_test(
        dataset_dir=dataset_dir,
        model_path=model_path
    )


