
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as transform
import cv2 as cv

# preprocess datasets
def adjust_data(img, label):
    if(np.max(img) > 1):
        img = img/255
        label = label/255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (img, label)

def generate_train(batch_size, train_path, image_folder, label_folder, aug_args,
                    save_dir = None, target_size = (424,424)):
    generate_image = ImageDataGenerator(**aug_args)
    generate_label = ImageDataGenerator(**aug_args)
    image_generator = generate_image.flow_from_directory(
        directory = train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgba",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_dir,
        save_prefix  = "image",
        seed = 1)
    label_generator = generate_label.flow_from_directory(
        directory = train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_dir,
        save_prefix  = "label",
        seed = 1)
    for (img,label) in zip(image_generator, label_generator):
        img,label = adjust_data(img,label)
        yield (img,label)

# preprocess test images
def generate_test(test_path, num_image = 10, image_size = (424, 424)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i))
        img = img/255
        img = transform.resize(img, image_size)
        img = np.expand_dims(img, axis = 0)
        yield img

# convert to numpy file for training
def generate_numpy(image_path,label_path):
    img_name= glob.glob(os.path.join(image_path,"%s*.png"%"image"))
    img_np_array = []
    label_np_array = []
    for index,item in enumerate(img_name):
        img = io.imread(item)
        img = img_to_array(img)
        label = io.imread(item.replace(image_path,label_path).replace("image","label"),as_gray = True)
        label = img_to_array(label)
        img,label = adjust_data(img,label)
        img_np_array.append(img)
        label_np_array.append(label)
    
    return img_np_array,label_np_array
