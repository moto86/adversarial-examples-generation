import tensorflow as tf
import keras
import numpy as np
import foolbox
import argparse
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

## parameter
parser = argparse.ArgumentParser()
parser.add_argument("-m",  "--model", required=True, type=str)
parser.add_argument("-a",  "--attack", required=True, type=str)
args = args = parser.parse_args()

model_file = args.model
attack = args.attack

## dataseet
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_shape = (28, 28, 1)
num_classes = 10
x_train = x_train.reshape(60000, 28, 28, 1)
x_test  = x_test.reshape(10000, 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /=255.0

## adversarial model
model = load_model(model_file)
fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))

if attack == 'fgsm':
    amodel = foolbox.attacks.FGSM(fmodel)
elif attack == 'cw':
    amodel = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
elif attack == 'deepfool':
    amodel = foolbox.attacks.DeepFoolAttack(fmodel)
else:
    print('wrong attack name')
    exit()

adv_imgs = np.zeros_like(x_test)
for index in range(x_test.shape[0]):
    label_array = np.where(y_test[index])
    label = label_array[0][0]
    if attack == 'fgsm':
        adv_img = amodel(x_test[index], label)
    elif attack == 'cw':
        adv_img = amodel(x_test[index], label)
    elif attack == 'deepfool':
        adv_img = amodel(x_test[index], label)
    adv_imgs[index] = adv_img

np.save(attack + '_' + model_file, adv_imgs)
