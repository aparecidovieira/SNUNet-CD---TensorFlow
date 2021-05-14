from utils.metrics import calculateIoUEpoch
import numpy as np
import cv2
import glob, shutil, os, sys
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard,Callback
from . import metrics as metrics

class Data_loader:
    def __init__(self, path_dataset, batch_size=8, change_detection=True):
        self.path_dataset = path_dataset + '/' if path_dataset[-1] != '/' else path_dataset
        self.batch_size = batch_size
        self.train_dataset = self.path_dataset + 'train/'
        self.val_dataset = self.path_dataset + 'val/'
        self.change_detection = change_detection
        self.train_filenames = [os.path.basename(name) for name in glob.glob(self.train_dataset + '*')]
        # self.val_filenames = [os.path.basename(name) for name in glob.glob(self.val_dataset + '*')]
        self.filenames = 0


    def get_labels(self, img, _type=True):
        # if _type:
        colors = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 255, 255): 1}
        h, w = img.shape[:2]
        img = self.roundColor_2D(img)
        mask = np.zeros((h, w), dtype=np.uint8)
        for color, value in colors.items():
            indexes = np.all(img == np.array(color).reshape(1, 1, 3), axis=2)
            mask[indexes] = value
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def get_image(self, path_img, dataset, masks=False):

        img = cv2.imread(path_img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if masks:
            img = self.get_labels(img[:, :h, :])
        else:
            img = np.float32(img) / 255.0

        
        return img


    def scale(self, image):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image

    def roundColor_2D(self, img):
        img[img > 70 ] = 255
        img[img <= 70 ] = 0
        return(img)


    def train_files_sample(self):
        self.filenames = [os.path.basename(name) for name in glob.glob(self.train_dataset + '*')][:]
        return len(self.filenames)

    def data_generator(self, dataset='train'):
        filepath = self.train_dataset if dataset == 'train' else self.val_dataset
        self.filenames = [os.path.basename(name) for name in glob.glob(filepath + '*')][:]
        print(len(self.filenames), 'Number of training images ~~~')
        images_files = itertools.cycle(zip(self.filenames))
        while True:
            batch_input = []
            batch_output = []
            for _ in range(self.batch_size):
                filename = next(images_files)[0]
                input_img = self.get_image(self.path_dataset + dataset + '/' + filename, dataset)
                output_img = self.get_image(self.path_dataset + dataset + '_labels/' + filename, dataset, masks=True) 
                batch_input.append(input_img)
                batch_output.append(output_img.astype(int))

            yield (np.array(batch_input), np.array(batch_output))


class onEachEpochCheckPoint(Callback):
    def __init__(self, model, path, val_inputs_path, checkpoint_dir, one_hot_label=False, height=512, width=512):
        super().__init__()
        self.path = path 
        self.model = model
        self.val_inputs_path = val_inputs_path
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self.model.save_weights(self.path.format(epoch), overwrite=True)
        _metrics = metrics.calculateIoUEpoch(self.model, self.val_inputs_path, self.checkpoint_dir, epoch)
        BG_IU, BD_IU, BG_P, BD_P = _metrics
        print("\nBackground IOU = %02f"%BG_IU)
        print("Main-Class IOU = %02f"%BD_IU)
        print("Mean IOU = %02f"%((BG_IU + BD_IU)/2))
        print("Background P-Accuracy = %02f"%BG_P)
        print("Main-Class P-Accuracy = %02f\n"%BD_P)


def parse_image(img_path:str) -> dict:

    image = tf.io.read_file(img_path)
    image = tf.io.decode_png(image, channels = 3)
    mask_path = tf.strings.regex_replace(img_path, 'train', 'train_labels')
    mask_path = tf.strings.regex_replace(mask_path, 'val', 'val_labels')
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}