import os
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

from utils.dataset import TfdataPipeline
from utils.losses import ContentLoss, StyleLoss
from utils.preprocess_io import preprocess_input
from model.style_transfer_net import StyleTransferNet

tf.random.set_seed(42)

def train(
    BASE_DATASET_DIR: str = 'style_content_dataset/',
    checkpoint_dir: str = 'trained_model/',
    pre_trained_model_dir: str = None,
    img_h: int = 256,
    img_w: int = 256,
    img_c: int = 3,
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate = 1e-3,
    loss_w:float = 0.5,
    encoder_model_arc:str = 'vgg',
    log_dir:str = 'logs/',
    saving_freq:int = 5,
):
    '''
    Main Train fucntion to train our Style Transfer Net.
    args:
        BASE_DATASET_DIR: str, the directory of the dataset
        checkpoint_dir: str, the directory to save the trained style encoder model
        pre_trained_model_dir: str, the directory of the pre-trained model to continue training
        img_h: int, the height of the image
        img_w: int, the width of the image
        img_c: int, the number of channels of the image
        batch_size: int, the batch size of the dataset
        epochs: int, the number of epochs to train
        learning_rate: float, the learning rate of the optimizer
        loss_w: float, the weight of the style loss
        encoder_model_arc: str, the architecture of the encoder model
        log_dir: str, the directory to save the tensorboard logs
    '''
    # Basic Sanity checks
    assert img_h > 224 and img_w > 224 and img_c == 3, 'The image size should be larger than 224x224x3.'
    assert os.path.exists(BASE_DATASET_DIR), f'The directory {BASE_DATASET_DIR} does not exist.'
    assert encoder_model_arc in ['vgg'], 'The encoder model should be vgg.'

    # Create checkpoint directory and log directory if they dont exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Instatiate tf.summary.FileWriter
    logdir = f'{log_dir}/StyleTransferNet_{encoder_model_arc}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    train_writer = tf.summary.create_file_writer(logdir+'/train/')

    # Instantiate dataset pipeline
    tf.print('Instantiating dataset pipeline...\n')
    dataset = TfdataPipeline(
        BASE_DATASET_DIR=BASE_DATASET_DIR,
        IMG_H=img_h,
        IMG_W=img_w,
        IMG_C=img_c,
        batch_size=batch_size,
    )
    train_ds = dataset.data_loader(dataset_type='train')

    # Instantiate Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Instantiate Losses
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # Instantiate Style Transfer Net
    tf.print('Checking for pre-trained model...\n')
    if pre_trained_model_dir is not None:
        if os.path.exists(pre_trained_model_dir):
            tf.print('Pre-trained model found! Loading...\n')
            nst_model = models.load_model(pre_trained_model_dir)
        else:
            tf.print(f'{pre_trained_model_dir} does not exist!\n')
    else:
        tf.print('Pre-trained model not found! Creating Style Transfer Net...\n')
        nst_model = StyleTransferNet(
            IMG_H=img_h,
            IMG_W=img_w,
            encoder_model_arc=encoder_model_arc
        )

    # Compile the model
    tf.print('Compiling the model...\n')
    nst_model.compile(
        optimizer=optimizer,
        content_loss=content_loss,
        style_loss=style_loss,
        loss_weight=loss_w,
    )
            

    tf.print(f'[INFO] Summary of model\n')
    nst_model.summary()

    tf.print('\n')
    tf.print('*'*60)
    tf.print('\t\t\tModel Configs')
    tf.print('*'*60)

    tf.print(
        f'\n',
        f'Model Name                : {nst_model.name}\n',
        f'Input shape               : ({img_h, img_w, img_c})\n',
        f'Epochs                    : {epochs}\n',
        f'Batch Size                : {batch_size}\n',
        f'Learning Rate             : {learning_rate}\n',
        f'Loss Weight               : {loss_w}\n',
        f'Encoder Model Architecture: {encoder_model_arc}\n',
        f'Saving Frequency          : {saving_freq}\n',
        f'\n',
    )

    # Training model
    tf.print('Training the model...\n')
    for e in range(epochs):
        try:
            t = time()
            avg_style_loss = []
            avg_content_loss = []
            for content_img, style_img in tqdm(train_ds, desc='Training...', unit='steps', colour='red'):
                # Prerocess the images before feeding to VGG19 model
                content_img = preprocess_input(content_img*255.0)
                style_img = preprocess_input(style_img*255.0)
                content_loss, style_loss = nst_model.train_step(content_img, style_img)
                style_loss = tf.math.log(style_loss)
                content_loss = tf.math.log(content_loss)
                avg_style_loss.append(style_loss)
                avg_content_loss.append(content_loss)
            
            avg_style_loss = tf.reduce_mean(avg_style_loss)
            avg_content_loss = tf.reduce_mean(avg_content_loss)
            tf.print(f"ETA:{round((time() - t)/60, 2)} - epoch: {(e+1)} - content_loss: {avg_content_loss} - style_loss: {avg_style_loss} \n")

            tf.print('Writing summary...\n')
            with train_writer.as_default():
                tf.summary.scalar('content_loss', avg_content_loss, step=e)
                tf.summary.scalar('style_loss', avg_style_loss, step=e)
            
            if (e+1) % saving_freq == 0:
                tf.print(f'Saving model at epoch {e+1}...\n')
                nst_model.save(f'{checkpoint_dir}/{nst_model.name}_iteration_{e+1}', save_format='tf')
                nst_model.decoder.save(f'{checkpoint_dir}/{nst_model.name}_decoder_iteration_{e+1}', save_format='tf')
                tf.print(f'Saved model at epoch {e+1}\n')
        except:
            tf.print(f'Error occured at epoch {e+1}\n')
            tf.print('Saving model at epoch {e+1}...\n')
            nst_model.save(f'{checkpoint_dir}/{nst_model.name}_iteration_{e+1}', save_format='tf')
            nst_model.decoder.save(f'{checkpoint_dir}/{nst_model.name}_decoder_iteration_{e+1}', save_format='tf')
            tf.print(f'Saved model at epoch {e+1}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='The directory of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='trained_model/', help='The directory to save the trained style encoder model')
    parser.add_argument('--pre_trained_model_dir', type=str, default=None, help='The directory of the pre-trained nst_model')
    parser.add_argument('--img_h', type=int, default=256, help='The height of the image')
    parser.add_argument('--img_w', type=int, default=256, help='The width of the image')
    parser.add_argument('--img_c', type=int, default=3, help='The number of channels of the image')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size of the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate of the optimizer')
    parser.add_argument('--loss_w', type=float, default=0.5, help='The weight of the style loss')
    parser.add_argument('--encoder_model_arc', type=str, default='vgg', help='The architecture of the encoder model')
    parser.add_argument('--log_dir', type=str, default='logs/', help='The directory to save the tensorboard logs')
    parser.add_argument('--saving_freq', type=int, default=2, help='The frequency of saving the trained model')
    args = parser.parse_args()

    train(
        BASE_DATASET_DIR=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        pre_trained_model_dir=args.pre_trained_model_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        img_c=args.img_c,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss_w=args.loss_w,
        encoder_model_arc=args.encoder_model_arc,
        log_dir=args.log_dir,
        saving_freq=args.saving_freq,
    )


if __name__ == "__main__":
    main()