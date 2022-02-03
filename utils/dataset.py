import os
from typing import Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob

import tensorflow as tf

from preprocess_io import preprocess_input, process_out

class TfdataPipeline:
    '''
    A class to create a tf.data.Dataset object from a directory of images.
    args:
        BASE_DATASET_DIR: str, the directory of the dataset
        IMG_H: int, the height of the image
        IMG_W: int, the width of the image
        IMG_C: int, the number of channels of the image
        batch_size: int, the batch size of the dataset
    '''
    def __init__(
        self,
        BASE_DATASET_DIR: str,
        IMG_H: int = 256,
        IMG_W: int = 256,
        IMG_C: int = 3,
        batch_size: int = 16,
    ) -> None:
        self.BASE_DATASET_DIR = BASE_DATASET_DIR
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C
        self.batch_size = batch_size
        self.__dataset_type = ['train', 'test']

        if not os.path.exists(self.BASE_DATASET_DIR):
            raise FileNotFoundError(
                f'The directory {self.BASE_DATASET_DIR} does not exist.'
            )
    
    def _load_image_file_names(self) -> list:
        '''
        Load the file names of the dataset.
        '''
        content_image_files = sorted(glob.glob(
            os.path.join(self.BASE_DATASET_DIR, 'content_images/*')
        ))
        style_image_files = sorted(glob.glob(
            os.path.join(self.BASE_DATASET_DIR, 'style_images/*')
        ))

        train_content_files = content_image_files[:int(len(content_image_files) * 0.99)]
        test_content_files = content_image_files[int(len(content_image_files) * 0.99):]

        train_style_files = style_image_files[:int(len(style_image_files) * 0.99)]
        test_style_files = style_image_files[int(len(style_image_files) * 0.99):]

        return (train_content_files, train_style_files), (test_content_files, test_style_files)
    
    # A funtion to load the image files with docstring
    def load_content_style_image(self, content_path:str, style_path:str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        Load the content and style images.
        args:
            content_path: str, the path of the content image
            style_path: str, the path of the style image
        return:
            content_image: tf.Tensor, the content image
            style_image: tf.Tensor, the style image
        '''
        
        # Read raw image from path
        content_image = tf.io.read_file(content_path)
        style_image = tf.io.read_file(style_path)

        # Decode the raw image
        content_image = tf.io.decode_image(content_image, channels=self.IMG_C, expand_animations=False)
        style_image = tf.io.decode_image(style_image, channels=self.IMG_C, expand_animations=False)

        # Change the dtype of the image to float32
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        style_image = tf.image.convert_image_dtype(style_image, tf.float32)

        # Resize the image to the desired size
        content_image = tf.image.resize_with_pad(content_image, self.IMG_H*2, self.IMG_W*2, method=tf.image.ResizeMethod.BICUBIC)
        style_image = tf.image.resize_with_pad(style_image, self.IMG_H*2, self.IMG_W*2, method=tf.image.ResizeMethod.BICUBIC)

    
        return content_image, style_image
    
    def _augment(self, x: tf.Tensor)-> tf.Tensor:
        '''
        Augment the images.
        args:
            x:tf.Tensor, Any image/tensor
        '''

        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        x = tf.image.random_crop(x, [self.IMG_H, self.IMG_W, self.IMG_C])
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

        return x
    
    def _tf_dataset(self, content_paths: list, style_paths:list, do_augment:bool)-> tf.data.Dataset:
        '''
        Creates a tf.data.Dataset object from the content and style image paths
        args:
            content_paths:list, the list of content image paths
            style_paths:list, the list of style image paths
            do_augment:bool, whether to augment the images
        '''
        # Create a tf.data.Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((content_paths, style_paths))

        # Map the function to load the content and style images
        dataset = (dataset
                    .map(lambda x,y : (self._augment(self.load_content_style_image(x,y)[0]), self._augment(self.load_content_style_image(x,y)[1])) if do_augment 
                        else (self.load_content_style_image(x,y)[0], self.load_content_style_image(x,y)[1]), 
                        num_parallel_calls=tf.data.AUTOTUNE)
                    # .cache()
                    .shuffle(buffer_size=20)
                    .batch(self.batch_size)
                    .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return dataset

    def data_loader(self, dataset_type:str = 'train', do_augment:bool = True)->tf.data.Dataset:
        '''
        Load the dataset.
        args:
            dataset_type:str, the type of the dataset, can be 'train' or 'test'
            do_augment:bool, whether to augment the images
        '''
        if dataset_type not in self.__dataset_type:
            raise ValueError(
                f'The dataset type {dataset_type} is not supported. '
                f'The supported dataset types are {self.__dataset_type}'
            )

        # Load the file names of the dataset
        if dataset_type == 'train':
            content_paths, style_paths = self._load_image_file_names()[0]
        else:
            content_paths, style_paths = self._load_image_file_names()[1]

        # Create a tf.data.Dataset object
        dataset = self._tf_dataset(content_paths, style_paths, do_augment)

        return dataset


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    conten_image_path = './style_content_dataset/content_images/0a0a615629231821.jpg'
    style_image_path = './style_content_dataset/style_images/10.jpg'
    batch_size = 1
    # Test the class
    tfdataset = TfdataPipeline(BASE_DATASET_DIR='./style_content_dataset/', batch_size=batch_size)

    train_ds = tfdataset.data_loader(dataset_type='train', do_augment=True)
    test_ds = tfdataset.data_loader(dataset_type='test', do_augment=True)

    for content, style_img in train_ds.take(1):
        content   = preprocess_input(content*255.0)
        style_img = preprocess_input(style_img*255.0)
        # content = content*255.0
        # style_img = style_img*255.0
        # tf.print(f'content shape: {content.shape}, style shape: {style_img.shape}')
        # tf.print(f'Max of content image: {tf.reduce_max(content)}')
        # tf.print(f'Max of style image: {tf.reduce_max(style_img)}')
        # tf.print(f'Min of content image: {tf.reduce_min(content)}')
        # tf.print(f'Min of style image: {tf.reduce_min(style_img)}')

        content = process_out(content)/255.0
        style_img = process_out(style_img)/255.0
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(content.numpy()[0])
        plt.subplot(1,2,2)
        plt.imshow(style_img.numpy()[0])
    plt.show()

