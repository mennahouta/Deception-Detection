#region imports
import glob
import keras, os
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Conv3D, MaxPool3D , Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf
#endregion

current_dir = os.getcwd()
dataset_path = current_dir + "/Dataset_VGG"

#region Functions' definitions.

def get_inner_paths(path):
    inner_paths = glob.glob(path + '/*')
    inner_paths.sort()
    inner_names = [os.path.basename(inner_path) for inner_path in inner_paths]
    return inner_paths, inner_names


def get_frames(video_path):
    frame_paths, _ = get_inner_paths(video_path)
    #frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    frames = []
    frame_num = 0
    while frame_num < len(frame_paths):
        frame_path = frame_paths[frame_num]
        frame = cv2.imread(frame_path)
        resized_frame = cv2.resize(frame, (112, 112))
        frames.append(resized_frame)
        frame_num += 10
    return frames


def read_dataset(what='train'):
    global dataset_path
    if what is 'train':
        path = dataset_path + '/train'
    else:
        path = dataset_path + '/test'
    video_paths, video_names = get_inner_paths(path)
    frames_per_video = [get_frames(video_path) for video_path in tqdm (video_paths)]
    return frames_per_video, video_names


def read_labels(what='train'):
    global dataset_path
    if what is 'train':
        path = dataset_path + '/train_labels.npy'
    else:
        path = dataset_path + '/test_labels.npy'
    return np.load(path, allow_pickle=True)

#endregion

#region Reading train videos and saving train numpys "Should run only once".
train_videos, train_video_names = read_dataset()
train_videos_np = np.array([np.array(video) for video in train_videos])
print(train_videos_np.shape)
print(type(train_videos_np[0]))
np.save(dataset_path + "/train_videos.npy", train_videos_np)
#endregion

#region Reading train numpy array "Run this EVERYTIME before training".
train_videos_np = np.load(dataset_path + '/train_videos.npy')
train_labels = read_labels()
#endregion

#region VGG Implementation

inpt_shpe = (144, 112, 112, 3)
model = Sequential()
#Block 1
model.add(Conv3D(64, (5, 3, 3), input_shape = inpt_shpe, padding='same', activation='relu', name='block1_conv1'))
model.add(Conv3D(64, (5, 3, 3), padding='same', activation='relu', name='block1_conv2') )
model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='block1_pool'))

#Block 2
model.add(Conv3D(128, (5, 3, 3), padding='same', activation='relu', name='block2_conv1'))
model.add(Conv3D(128, (5, 3, 3), padding='same', activation='relu', name='block2_conv2' ))
model.add(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name='block2_pool'))

#Block 3
model.add(Conv3D(256, (5, 3, 3), padding='same', activation='relu', name='block3_conv1'))
model.add(Conv3D(256, (5, 3, 3), padding='same', activation='relu', name='block3_conv2' ))
model.add(Conv3D(256, (5, 3, 3), padding='same', activation='relu', name='block3_conv3'))
model.add(Conv3D(256, (5, 3, 3), padding='same', activation='relu', name='block3_conv4') )
model.add(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name='block3_pool'))

#Block 4
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block4_conv1'))
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block4_conv2') )
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block4_conv3'))
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block4_conv4') )
model.add(MaxPool3D(pool_size=(1, 2, 2),strides=(1, 2, 2), name='block4_pool'))

#Block 5
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block5_conv1'))
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block5_conv2') )
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block5_conv3'))
model.add(Conv3D(512, (5, 3, 3), padding='same', activation='relu', name='block5_conv4') )
model.add(MaxPool3D(pool_size=(1, 2, 2),strides=(1, 2, 2), name='block5_pool'))

# Classification block
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(105, activation='softmax'))

model.summary()

opt = Adam(lr = 0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

#hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
#hist = model.fit(trainX, trainY, batch_size=32, epochs=100)
hist = model.fit(train_videos_np, train_labels, batch_size=32, epochs=100,  callbacks=[checkpoint,early])

#endregion