import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
import time, datetime
import argparse
import random
import os, sys, shutil, glob
import subprocess
from tensorflow.keras import losses
import tensorflow.keras.metrics as Metrics

# tf.enable_v2_behavior()

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# from tensorflow.python.client import device_lib
from utils.data_loader import  Data_loader
# from utils import model_loader
from models import siamese
# import helpers
# import utils_ as utils
# from utils_ import get_model
# import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# config.gpu_options.allow_growth = True
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--save', type=int, default=4, help='Interval for saving weights')
parser.add_argument('--gpu', type=str, default='0', help='Choose GPU device to be used')
parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--checkpoint', type=str, default="checkpoint", help='Checkpoint folder.')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default='AIRBUS', help='Dataset you are using.')
parser.add_argument('--load_data', type=str2bool, default=True, help='Dataset loading type.')
parser.add_argument('--act', type=str2bool, default=True, help='True if sigmoid or false for softmax')
parser.add_argument('--crop_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=200, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change.')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle.')
parser.add_argument('--model', type=str, default="dunet", help='The model you are using. Currently supports:\
     FPN, RFPN, Siamese')


args = parser.parse_args()
gpu = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES']=  gpu


# g = v1.Graph()

# with g.as_default():
# inputs = v1.placeholder(dtype=v1.float32, shape=(4))
# outputs = v1.placeholder(dtype=v1.float32, shape=(4))

def load_datasets(path_dataset, batch_size):
    datasets = Data_loader(path_dataset, batch_size=batch_size)
    return datasets.data_generator(), datasets.data_generator(dataset='val')

def load_info():
    dataset_path = args.dataset
    batch_size = args.batch_size
    train_dataset, val_dataset = load_datasets(dataset_path, batch_size)

    return train_dataset, val_dataset

def IoU(y_true, y_pred):
    def function(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        iou = (intersection + 1e-15)/(union + 1e-15)
        iou = iou.astype(np.float32)
        return iou
    return tf.numpy_function(function, [y_true, y_pred], tf.float32)

def iou(y_true, y_pred):
     def f(y_true, y_pred):
         intersection = (y_true * y_pred).sum()
         union = y_true.sum() + y_pred.sum() - intersection
         x = (intersection + 1e-15) / (union + 1e-15)
         x = x.astype(np.float32)
         return x
     return tf.numpy_function(f, [y_true, y_pred], tf.float32)

model_name = args.model
checkpoint_path = args.checkpoint
train_data, val_data = load_info()
n_epochs = args.num_epochs
# model = model_loader.get_model(model_name)
model = siamese.Siamese(num_classes=1,  input_shape=( None, None, 3))
## Metrics
loss_metric = tf.keras.metrics.Mean(name='train_loss')
# accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy"
)

optimizer = tf.keras.optimizers.Adam(0.0001)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(inputs)
        labels = tf.convert_to_tensor(labels)
        predictions = model(inputs, training=True)

        regularization_loss = tf.math.add_n(model.losses)
        pred_loss = loss_fn(labels, predictions)
        total_loss = pred_loss + regularization_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    ## Update the metrics
    loss_metric.update_state(total_loss)

save_path = checkpoint_path + '/' + model_name
dataset_path = args.dataset
def parse_image(img_path:str) -> dict:

    image = tf.io.read_file(img_path)
    image = tf.io.decode_png(image, channels = 3)
    # image = tf.image.convert_image_dtype(image, tf.unint8)
    mask_path = tf.strings.regex_replace(img_path, 'train', 'train_labels')
    mask_path = tf.strings.regex_replace(mask_path, 'val', 'val_labels')
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}
SEED = 42

train_dataset = tf.data.Dataset.list_files(dataset_path + 'train/' + "*.png", seed=SEED)
train_dataset = train_dataset.map(parse_image)
# dataset_path = args.dataset
val_dataset = tf.data.Dataset.list_files(dataset_path + 'val/' + "*.png", seed=SEED)
val_dataset =val_dataset.map(parse_image)
# train_input_names = train_data.train_filenames
dataset = {'train': train_dataset, 'val':val_dataset}


metrics_list = ['accuracy', iou]
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics_list)

model.fit(train_data, epochs=args.num_epochs)
loss, acc = model.evaluate(test_data)

print("Loss {}, Accuracy {}".format(loss, acc))


# for epoch in range(1):
#     id_list = 32#len(train_dataset)
#     num_iters = int(np.floor((id_list) / args.batch_size))

#     for _ in range(num_iters):
#         inputs, labels = next(train_data)

#             # print(inputs.shape, labels.shape)
#         train_step(inputs, labels)
#         # train_data.next()
#     print('Done epoch here', epoch)
# model.save( save_path+ '.h5')
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(train_data, epochs=1)
# loss, acc = model.evaluate(val_data)

# print("Loss {}, Accuracy {}".format(loss, acc))