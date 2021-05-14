import numpy as np
import cv2, shutil, os
from glob import glob
from . import data_loader as dl
import numpy as np


def get_labels(img, _type=True):
    # if _type:
    colors = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 255, 255): 1}
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for color, value in colors.items():
        indexes = np.all(img == np.array(color).reshape(1, 1, 3), axis=2)
        mask[indexes] = value
    mask = np.expand_dims(mask, axis=-1)
    return mask

def change_labels(img):
    h, w = img.shape[:2]
    colors = [(0, 0, 0), (255, 255, 255)]
    seg_img = np.zeros((h, w, 3))

    for i in range(len(colors)):
        seg_img[:, : , 0] =  ((img[:, :] == i) * colors[i][0]).astype('uint8')
        seg_img[:, : , 1] =  ((img[:, :] == i) * colors[i][1]).astype('uint8')
        seg_img[:, : , 2] =  ((img[:, :] == i) * colors[i][2]).astype('uint8')
    return seg_img



def get_image(path_img, masks=False):

    img = cv2.imread(path_img, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.float32(img) / 255.0
    h, w = img.shape[:2]
    if masks:
        img = get_labels(img[:, :h, :])
    else:
        img = np.float32(img) / 255.0

    return img

def get_IoU(y_pred, y_true, _class):
    tp = np.count_nonzero(np.logical_and(y_pred == _class, y_true == _class))
    fp = np.count_nonzero(np.logical_and(y_pred == _class, y_true != _class))
    tn = np.count_nonzero(np.logical_and(y_pred != _class, y_true != _class))
    fn = np.count_nonzero(np.logical_and(y_pred != _class, y_true == _class))
    return (tp, fn, fp, tn)


def calculate_Iou(tp, fp, fn):
    try:
        return float(sum(tp))/(sum(tp) + sum(fp) + sum(fn))
    except ZeroDivisionError:
        return 0

def calculate_pixelAcc(tp, fn):
    try : 
        return float(sum(tp))/(sum(tp) + sum(fn))
    except ZeroDivisionError:
        return 0

def calculateIoUEpoch(model, inputs_vals, checkpoint_dir, epoch_number):
    val_files = [imgPath for imgPath in glob(inputs_vals + '*')][:]

    print('Files for validation : ', len(val_files))
    os.makedirs(checkpoint_dir, exist_ok=True)
    classes = np.array([0, 1])
    save_path = "%s/epoch/%s/"%(checkpoint_dir, epoch_number) 
    os.makedirs(save_path, exist_ok=True)

    for filename in val_files:
        name = os.path.basename(filename)
        input_image = get_image(filename)
        mask = get_image(filename.replace('val/', 'val_labels/'), masks=True)[:, :, 0].astype(int)
        pred = model.predict(np.expand_dims(input_image, axis=0), batch_size=None, verbose=0, steps=None)
        pred = np.round(pred[0,:,:,0]).astype(int)      #(pred *255).astype(int)
        bg_out, class_out = [], []
        for _class in classes:
            out = get_IoU(pred, mask, _class)
            if _class == 0:
                bg_out.append(out)
            else:
                class_out.append(out)
        pred, mask = change_labels(pred), change_labels(mask)
        img_out = np.concatenate((input_image * 255, mask, pred), axis=1)
        cv2.imwrite(save_path + name, img_out)
    class_out = np.array(class_out)
    TP, FN, FP, TN = [list(class_out[:, i]) for i in range(4)]
    class_iou = 100 * calculate_Iou(TP, FP, FN)
    class_acc = 100 * calculate_pixelAcc(TP, FN)

    bg_out = np.array(bg_out)

    TP, FN, FP, TN = [list(bg_out[:, i]) for i in range(4)]
    bg_iou = 100 * calculate_Iou(TP, FP, FN)
    bg_acc = 100 * calculate_pixelAcc(TP, FN)

    return bg_iou, class_iou, bg_acc, class_acc