import csv
import cv2
from datetime import datetime
import json
import numpy as np
from pprint import pprint
from sklearn.utils import shuffle

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2


np.random.seed(1337)

log_path = './50Hz/master_50hz_driving_log.csv'

with open(log_path, 'r') as filey:
    log = list(csv.reader(filey)) #
    log = log[1:]
    print(len(log))
    
    np.random.shuffle(log) #SPLIT INTO TRAIN AND TEST

    test_cutoff = int(np.floor(.85*len(log)))
    validation_cutoff = int(np.floor(.80*test_cutoff))

    test_data = log[test_cutoff:]
    validation_data = log[validation_cutoff:test_cutoff]
    train_data = log[:validation_cutoff]

    print(len(test_data), len(validation_data), len(train_data))


def fixulate_image_from_file(filename):
    # read image and force RGB
    stripped = filename.strip()
    image = cv2.imread(stripped, 1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    return image


def augment_affine_translation(image, label, x_trans_perc=.10, y_trans_perc=.10):
    y_trans_px = y_trans_perc * image.shape[0]
    x_trans_px = x_trans_perc * image.shape[1]
    
    y_trans = np.random.uniform(-y_trans_px, y_trans_px)
    x_trans = np.random.uniform(-x_trans_px, x_trans_px)
    
    affine_M = np.float32([[1,0,x_trans],[0,1,y_trans]])
    
    augmented_image = cv2.warpAffine(image, affine_M, (image.shape[1], image.shape[0]))
    augmented_label = label + (x_trans * .012)
    return augmented_image, augmented_label


def augment_overall_brightness(image, label):
    brightness_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    
    if np.random.uniform() <= 0.5:
        brightness_image[:,:,2] = brightness_image[:,:,2] * np.random.uniform()
    else:
        brightness_image[:,:,2] = brightness_image[:,:,2] + ((1-brightness_image[:,:,2]) * np.random.uniform())

    augmented_image = cv2.cvtColor(brightness_image,cv2.COLOR_HSV2RGB)
    return augmented_image, label


def augment_insert_linear_shadow(image, label):
    # double mgrid method for creating mask adapted very closely from Vivek's blog
    shadow_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    
    y0, y1 = np.floor(320*np.random.uniform(size = 2))
    shadow_mask = np.ones(shadow_image[:,:,1].shape, dtype=np.float64)
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[((X_m * (y1 - y0)) - (shadow_image.shape[0] * (Y_m - y0)) >= 0)] = np.random.uniform(.1, .8)
    if np.random.uniform() < 0.5: shadow_mask = np.fliplr(shadow_mask)
    shadow_image[:,:,2] = shadow_image[:,:,2] * shadow_mask
    
    augmented_image = cv2.cvtColor(shadow_image, cv2.COLOR_HSV2RGB)
    return augmented_image, label

def reshape_image(image):
    res = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    height = res.shape[0]
    res = res[height*0.15:,:,:]
    return res

def generate_image_data(data_log, batch_size=128):
    cam_angle = 0.28
    cam_info = [(0, 0),(1, cam_angle),(2, 0-cam_angle)]

    while True:
        i = 0
        X_batch, y_batch = list(), list()
        while i < batch_size:
            rand0, rand1, rand2, rand3, rand4 = np.random.random(5)
            line = data_log[np.random.randint(0, len(data_log))]
            original_angle = np.float64(line[3])
            if np.abs(original_angle) > 0.05 or rand0 > .95:
                cam_choice = np.random.choice([0,0,1,2])
                image = fixulate_image_from_file(line[cam_info[cam_choice][0]])
                angle = original_angle + cam_info[cam_choice][1]
                if rand1 <= 0.5:
                    angle, image = 0-angle, cv2.flip(image, 1)
#                 if rand2 <= 0.5:
#                     image, angle = augment_affine_translation(image, angle)
                if rand3 <= 0.5:
                    image, angle = augment_overall_brightness(image, angle)
                if rand4 <= 0.5:
                    image, angle = augment_insert_linear_shadow(image,angle)
#                 blur = cv2.bilateralFilter(image, 9, 75, 75) # denoise without harming edges
                blur =  image# denoise without harming edges
                res = reshape_image(blur)
                X_batch.append(res)
                y_batch.append(angle)
                i += 1
        yield (np.array(X_batch), np.array(y_batch))
        

X_val = np.array([reshape_image(fixulate_image_from_file(x[0])) for x in test_data])
y_val = np.array([x[3] for x in test_data])
X_test = np.array([reshape_image(fixulate_image_from_file(x[0])) for x in validation_data])
y_test = np.array([x[3] for x in validation_data])

index = 1        

input_shape = X_val[0].shape
print(input_shape)

model = Sequential()

pool_dim = (2,2)
stride_dim = (2,2)

model.add(Lambda(lambda x: (x / 255.) - .5,
                     input_shape=input_shape))

filters = 3
kernel_dim = 1
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='same'))

filters = 24
kernel_dim = 5
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='valid', subsample=stride_dim, activation='relu'))
model.add(Dropout(0.25))


filters = 32
kernel_dim = 5
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='valid', subsample=stride_dim, activation='relu'))
model.add(Dropout(0.25))

filters = 48
kernel_dim = 5
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='valid', subsample=stride_dim, activation='relu'))
model.add(Dropout(0.25))

filters = 64
kernel_dim = 3
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='valid', activation='relu'))


filters = 64
kernel_dim = 3
model.add(Convolution2D(filters, kernel_dim, kernel_dim,
                        border_mode='valid', activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


batch_size = 2*128
model.fit_generator(generate_image_data(train_data, batch_size), samples_per_epoch=batch_size * 100, nb_epoch=5, validation_data=(X_val, y_val), callbacks=[], class_weight=None, nb_worker=1)
score = model.evaluate(X_test, y_test, batch_size)


print('Test score:', score[0])
print('Test accuracy:', score[1])


json_string = model.to_json()
with open('./model.json', 'w') as filey:
    filey.write(json_string)
with open('./BACKUP_model.json_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), 'w') as filey:
    filey.write(json_string)


model.save_weights('model.h5')
model.save_weights('BACKUP_model.h5_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))





