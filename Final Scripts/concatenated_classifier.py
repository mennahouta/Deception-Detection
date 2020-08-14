# region Imports
import cv2
import glob

from tqdm import tqdm

from keras import Model
import keras
from keras.layers import (Conv3D, Dense, Flatten, MaxPooling3D, Lambda)

from keras.layers.merge import concatenate
from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from numpy import save
from numpy import load
import scipy
from scipy import ndimage
from sklearn.metrics import accuracy_score
# endregion

# region Constants
DIM_WIDTH = 40
DIM_HEIGHT = 60
FLAG_TO_GRAY = False
# endregion


# region Helper Functions
def get_volumes_micro_exps(micro_exps, num_of_volumes):
    """
    Original shape: (692, 39)
    N = 692 * num_of_volumes
    Transformed shape: [(N, 13, 1, 1, 1), (N, 13, 1, 1, 1), (N, 13, 1, 1, 1)]
    So they can be concatenated with the other FC inputs which are of shapes (None, 13, 1, 1, 1)
    """
    volumes_micro_expressions = []
    for micro_expressions in micro_exps:
        for i in range(num_of_volumes):
            volumes_micro_expressions.append(micro_expressions)
    volumes_micro_expressions = np.asarray(volumes_micro_expressions, dtype='float32')
    print("Volumes' Micros shape: ", volumes_micro_expressions.shape)
    micro_exps_first_13 = volumes_micro_expressions[:, :13].reshape(-1, 13, 1, 1, 1)
    micro_exps_second_13 = volumes_micro_expressions[:, 13:13*2].reshape(-1, 13, 1, 1, 1)
    micro_exps_third_13 = volumes_micro_expressions[:, 13*2:].reshape(-1, 13, 1, 1, 1)
    return [micro_exps_first_13, micro_exps_second_13, micro_exps_third_13]


def get_clips_paths(folder_path: str, extension: str) -> (list, list):
    """
    ## Build the Clips into Lists
    * This function takes the path of either the Deceptive data or the Truthful.
    * It returns a list of videos.
    * Each list of videos contains a list of clips.
    * Each list of clips contain the frames' paths.
    """
    if folder_path[-1] != '/':
        folder_path += '/'
    if extension[0] != '*':
        extension = "*" + extension

    vids_paths = []
    vids_clips_number = []
    for trial_f_num in tqdm(range(1, 121 + 1)):
        # for ranges lie [1, 61]
        trial_name = "lie" + format(trial_f_num, '03d')
        # for ranges truth [1, 60]
        if trial_f_num > 61:
            trial_name = "truth" + format(trial_f_num - 61, '03d')
        trial_folder = "*" + trial_name + '/'
        clip_paths = []
        for sub_f_num in range(25):
            sub_folder = format(sub_f_num, '03d') + '/'
            paths = sorted(glob.glob(folder_path + trial_folder + sub_folder + extension))
            if len(paths) == 0:
                continue
            clip_paths.append(paths)
        # If no subfolders were found, try searching for the files in the folder
        if len(clip_paths) == 0:
            paths = sorted(glob.glob(folder_path + trial_folder + extension))
            if len(paths) == 0:
                continue
            clip_paths.append(paths)
        # if no files were found in the whole folder, continue
        if len(clip_paths) == 0:
            continue
        vids_paths.append(clip_paths)
        vids_clips_number.append(len(clip_paths))
    return vids_paths, vids_clips_number


def read_cropped_frames(clip_paths: list, resize_shape: tuple = (40, 60), f_convert=True):
    clip = []
    for i in range(0, len(clip_paths), 4):
        clip_path = clip_paths[i]
        cur_frame = cv2.imread(clip_path)
        # Convert the RBG frame to gray-scale.
        if f_convert:
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        # resizing the frame to (60 * 40)
        cur_frame = cv2.resize(cur_frame, resize_shape)
        # write the converted gray frame into the output video.
        cur_frame_arr = np.array(cur_frame.tolist(), dtype='uint8')
        clip.append(cur_frame_arr)
    cv2.destroyAllWindows()
    return clip


def to_one_hot(data_y):
    data_hot = np.zeros((data_y.shape[0], 2), dtype='uint8')
    data_hot[:, 0] = (data_y == 0)
    data_hot[:, 1] = (data_y == 1)
    return data_hot
# endregion


# region Model Architecture
def crop(dimension, start, end):
    """
    Crops (or slices) a Tensor on a given dimension from start to end
    example : to crop tensor x[:, :, 5:10]
    call slice(2, 5, 10) as you want to crop on the third dimension
    """
    def func(volume):
        if dimension == 0:
            return volume[start: end]
        if dimension == 1:
            return volume[:, start: end]
        if dimension == 2:
            return volume[:, :, start: end]
        if dimension == 3:
            return volume[:, :, :, start: end]
        if dimension == 4:
            return volume[:, :, :, :, start: end]
    return Lambda(func)


def H1(cur_volume: np.ndarray(shape=(7, 60, 40))):
    """
    ## Hard-wired Layer

    * Take steps of 7 frames.
        * Copy volume of gray frames. *(7 feature maps)*
        * Apply gradient filter on the x-axis per frame. *(7 feature maps)*
        * Apply gradient filter on the y-axis per frame. *(7 feature maps)*
        * Extract optical flow channels for each 2 consecutive frames in both x and y. *(6 feature maps each)*
        * Concatenate the resulting volumes, 5 channels.

    **Input dimensions (7@60x40)**

    **Output dimensions (33@60x40)**, 33 feature maps.
    """
    volume = cur_volume
    prev_frame = None
    gx_vol = None
    gy_vol = None
    optx_vol = None
    opty_vol = None
    for frame in cur_volume:
        # For each frame, calculate the frame gradients

        # Using the scipy library to get the image gradient in the x-axis.
        g_x = ndimage.sobel(frame, axis=0, mode='constant')
        # Reshape the numpy array to have the channels first.
        g_x = np.reshape(g_x, (1, g_x.shape[0], g_x.shape[1]))
        # Append the current x_gradient to the output volume.
        if gx_vol is not None:
            gx_vol = np.concatenate((gx_vol, g_x), axis=0)
        else:
            gx_vol = g_x

        # Using the scipy library to get the image gradient in the y-axis.
        g_y = ndimage.sobel(frame, axis=1, mode='constant')
        # Reshape the numpy array to have the channels first.
        g_y = np.reshape(g_y, (1, g_y.shape[0], g_y.shape[1]))
        # Append the current y_gradient to the output volume.
        if gy_vol is not None:
            gy_vol = np.concatenate((gy_vol, g_y), axis=0)
        else:
            gy_vol = g_y

        if prev_frame is not None:
            # Using the cv2 built-in function to get the optical flow map,
            # given the previous frame, the current one.
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Output is a numpy array with 2 channels,

            # [0] is the optical flow map on the x-axis
            # Reshape the output optical map on the to have the channels first.
            flow_x = np.reshape(flow[:, :, 0], (1, 60, 40))
            if optx_vol is not None:
                optx_vol = np.concatenate((optx_vol, flow_x), axis=0)
            else:
                optx_vol = flow_x

            # [1] is the optical flow map on the y-axis
            flow_y = np.reshape(flow[:, :, 1], (1, 60, 40))
            if opty_vol is not None:
                opty_vol = np.concatenate((opty_vol, flow_y), axis=0)
            else:
                opty_vol = flow_y
        prev_frame = frame
    cv2.destroyAllWindows()
    volume = volume.astype('uint8')
    gx_vol = gx_vol.astype('int64')
    gy_vol = gy_vol.astype('int64')
    optx_vol = optx_vol.astype('float64')
    opty_vol = opty_vol.astype('float64')
    volume = np.concatenate((volume, gx_vol, gy_vol, optx_vol, opty_vol), axis=0)

    return volume


def C2(input):
    """
    ## First Convolutional Layer
    * On each channel apply a kernal **(7x7x3)** twice.
    *(7x7 in spatial domain, 3 in the temporal)

    **Input (33@60x40)**

    **Output (2*23@54x34)**
    """
    # kernel_size(depth, (kernel_shape))
    gray   = Conv3D(1, activation='tanh', kernel_size=(3, 7, 7), padding='valid',
                 input_shape=(7, 60, 40, 1), data_format='channels_last')(crop(1, 0, 7)(input))

    grad_x = Conv3D(1, activation='tanh', kernel_size=(3, 7, 7), padding='valid',
                 input_shape=(7, 60, 40, 1), data_format='channels_last')(crop(1, 7, 14)(input))

    grad_y = Conv3D(1, activation='tanh', kernel_size=(3, 7, 7), padding='valid',
                 input_shape=(7, 60, 40, 1), data_format='channels_last')(crop(1, 14, 21)(input))

    opt_x  = Conv3D(1, activation='tanh', kernel_size=(3, 7, 7), padding='valid',
                 input_shape=(6, 60, 40, 1), data_format='channels_last')(crop(1, 21, 27)(input))

    opt_y  = Conv3D(1, activation='tanh', kernel_size=(3, 7, 7), padding='valid',
                 input_shape=(6, 60, 40, 1), data_format='channels_last')(crop(1, 27, 33)(input))

    # concatencation of the 5 channels is done on the second/dim-1
    # which is the dimenion of frames, all are put in one set.
    out_set = concatenate([gray, grad_x, grad_y, opt_x, opt_y], axis=1)
    return out_set


def S3(input):
    """
    ## Apply subsampling **(max-pooling 2x2)**
    **Output (2*23@27x17)**
    """
    # to conserve the depth, to not be affected by the downsampling, we but it = 1.
    out_set = MaxPooling3D(pool_size=(1, 2, 2))(input)
    return out_set


def C4(input):
    """
    ## Second Convolutional Layer
    * On each channel apply a kernal **(7x6x3)** thrice.

    **Output (6*13@21x12)**
    """
    # kernel_size(depth, (kernel_shape))
    gray   = Conv3D(1, activation='tanh', kernel_size=(3, 7, 6), padding='valid',
                 input_shape=(5, 27, 17, 1), data_format='channels_last')(crop(1, 0, 5)(input))

    grad_x = Conv3D(1, activation='tanh', kernel_size=(3, 7, 6), padding='valid',
                 input_shape=(5, 27, 17, 1), data_format='channels_last')(crop(1, 5, 10)(input))

    grad_y = Conv3D(1, activation='tanh', kernel_size=(3, 7, 6), padding='valid',
                 input_shape=(5, 27, 17, 1), data_format='channels_last')(crop(1, 10, 15)(input))

    opt_x  = Conv3D(1, activation='tanh', kernel_size=(3, 7, 6), padding='valid',
                 input_shape=(4, 27, 17, 1), data_format='channels_last')(crop(1, 15, 19)(input))

    opt_y  = Conv3D(1, activation='tanh', kernel_size=(3, 7, 6), padding='valid',
                 input_shape=(4, 27, 17, 1), data_format='channels_last')(crop(1, 19, 23)(input))

    # concatencation of the 5 channels is done on the second/dim-1
    # which is the dimenion of frames, all are put in one set.
    out_set = concatenate([gray, grad_x, grad_y, opt_x, opt_y], axis=1)
    return out_set


def S5(input):
    """
    ## Apply subsampling **(max-pooling 3x3)**

    ** Output(6 * 13 @ 7x4) **
    """
    output = MaxPooling3D(pool_size=(1, 3, 3))(input)
    return output


def C6(input):
    """
    ## Third Convolutional Layer
    * On each channel apply a kernal **(7x4)**

    **Output (78@1x1)**
    """
    # kernel_size(depth, (kernel_shape))
    gray   = Conv3D(1, activation='tanh', kernel_size=(1, 7, 4), padding='valid',
                 input_shape=(3, 7, 4, 1), data_format='channels_last')(crop(1, 0, 3)(input))

    grad_x = Conv3D(1, activation='tanh', kernel_size=(1, 7, 4), padding='valid',
                 input_shape=(3, 7, 4, 1), data_format='channels_last')(crop(1, 3, 6)(input))

    grad_y = Conv3D(1, activation='tanh', kernel_size=(1, 7, 4), padding='valid',
                 input_shape=(3, 7, 4, 1), data_format='channels_last')(crop(1, 6, 9)(input))

    opt_x  = Conv3D(1, activation='tanh', kernel_size=(1, 7, 4), padding='valid',
                 input_shape=(2, 7, 4, 1), data_format='channels_last')(crop(1, 9, 11)(input))

    opt_y  = Conv3D(1, activation='tanh', kernel_size=(1, 7, 4), padding='valid',
                 input_shape=(2, 7, 4, 1), data_format='channels_last')(crop(1, 11, 13)(input))
    # concatencation of the 5 channels is done on the second/dim-1
    # which is the dimenion of frames, all are put in one set.
    out_set = concatenate([gray, grad_x, grad_y, opt_x, opt_y], axis=1)
    return out_set


def FC(input:list):
    """
    ## FC-Layer of **(128@1x1)**
    * Connect to output layer.
    """
    # input: a list of sets representing the last output from layer C6.
    output = Flatten()(input)
    output = Dense(units=128, activation='tanh')(output)
    return output


# without hardwired included
def ConcatenatedModel():
    input = keras.layers.Input(shape=(33, 60, 40, 1))
    micro_exps0 = keras.layers.Input(shape=(13, 1, 1, 1))
    micro_exps1 = keras.layers.Input(shape=(13, 1, 1, 1))
    micro_exps2 = keras.layers.Input(shape=(13, 1, 1, 1))
    # 1st ConvLayer
    C2_set1 = C2(input)
    C2_set2 = C2(input)
    # Subsampling
    C2_set1 = S3(C2_set1)
    C2_set2 = S3(C2_set2)
    # 2nd ConvLayer
    C4_set1 = C4(C2_set1)
    C4_set2 = C4(C2_set1)
    C4_set3 = C4(C2_set1)

    C4_set4 = C4(C2_set2)
    C4_set5 = C4(C2_set2)
    C4_set6 = C4(C2_set2)
    # Subsampling
    C4_set1 = S5(C4_set1)
    C4_set2 = S5(C4_set2)
    C4_set3 = S5(C4_set3)

    C4_set4 = S5(C4_set4)
    C4_set5 = S5(C4_set5)
    C4_set6 = S5(C4_set6)
    # 3rd ConvLayer
    C6_set1 = C6(C4_set1)
    C6_set2 = C6(C4_set2)
    C6_set3 = C6(C4_set3)
    C6_set4 = C6(C4_set4)
    C6_set5 = C6(C4_set5)
    C6_set6 = C6(C4_set6)
    # Fully-connected layer
    to_conc_list = [C6_set1, C6_set2, C6_set3, C6_set4, C6_set5, C6_set6, micro_exps0, micro_exps1, micro_exps2]
    FC_input = concatenate(to_conc_list, axis=1)

    FC_out = FC(FC_input)
    # Output Layer
    output = Dense(units=2, activation='softmax')(FC_out)
    return Model(inputs=[input, micro_exps0, micro_exps1, micro_exps2], outputs=output)


def test(model, x_test, y_test, clips_mapping, micros_test):
    """
    A function that inputs the clips and their actual prediction
    and the mapping of the videos to the clips. The videos'
    prediction will be through majority voting on their clips.

    Parameters
    ----------
    model : tensorflow Model()
    x_test : np.ndarray(shape=(None, 33, 60, 40, 1))
        The clips to be tested. The shape indicates the product of
        #volumes_per_clip and #clips, the number of feature maps per
        volume, then the spatial dimensions (60, 40, 1).
    y_test : np.ndarray(shape=(None, 2))
        The actual prediction of each cell. The shape indicates the
        product of #volumes_per_clip and #clips, and the prediction
        which is either [1, 0]->Deceptive or [0, 1]->Truthful
    clips_mapping : np.ndarray(shape=(None, 2))
        The mapping of each video to the clips it's divided into.
        The shape indicates the number of videos to be tested, and
        each row->[x, y]. Where x: number of clips, y: number of
        volumes per clip.
    """
    # Get the prediction of each video
    vid_test = np.zeros((clips_mapping.shape[0], 2), dtype='int8')
    vid_pred = np.zeros((clips_mapping.shape[0], 2), dtype='int8')
    prev_batch_index = 0
    for i in range(clips_mapping.shape[0]):
        num_clips = clips_mapping[i, 0]
        num_volumes = clips_mapping[i, 1]
        num_cells = num_clips * num_volumes
        y_pred = model.predict([x_test[prev_batch_index:(prev_batch_index + num_cells)],
                                micros_test[0][prev_batch_index:(prev_batch_index + num_cells)],
                                micros_test[1][prev_batch_index:(prev_batch_index + num_cells)],
                                micros_test[2][prev_batch_index:(prev_batch_index + num_cells)]
                                ])
        classes = np.argmax(y_pred, axis=1)
        class_pred = 0  # Deceptive
        if np.sum(classes)*2 >= classes.shape[0]:
            class_pred = 1  # Truthful
        vid_pred[i, class_pred] = 1
        vid_test[i] = y_test[prev_batch_index]
        prev_batch_index += num_cells
    # Calculate the accuracy, the predicted value compared to the actual
    test_accuracy = accuracy_score(y_true=np.argmax(vid_test, axis=1), y_pred=np.argmax(vid_pred, axis=1))
    return test_accuracy, np.argmax(vid_pred, axis=1), np.argmax(vid_test, axis=1)
# endregion


def build_model():
    model = ConcatenatedModel()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    file_path = "weights.best.hdf5"
    mcp_save = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4,
                                       mode='min')
    callbacks_lst = [early_stopping, mcp_save, reduce_lr_loss]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model, callbacks_lst


def H1_preprocessing(input_videos, video_label, mapping, micro_expressions):
    """
    Example on micro_expressions:
    micro_expressions = load("/content/gdrive/My Drive/Team's Drive/3DCNN Numpys/test_micro_exps.npy")
    """
    step = 6
    inside_step = 2

    # drunkly verified
    num_of_volumes = ((input_videos.shape[1] - (step * inside_step)) // step) + 1
    # train_data_shape(number videos, number volumes)
    train_data_shape = (input_videos.shape[0], num_of_volumes)
    # N: "number videos" * "number volumes"
    N = train_data_shape[0] * train_data_shape[1]
    data_x = np.empty((train_data_shape[0], train_data_shape[1], 33, 60, 40), dtype='float64')
    data_y = np.zeros((train_data_shape[0], train_data_shape[1], 2), dtype='int8')

    vid_mapping = np.empty((mapping.shape[0], 2), dtype='int32')
    vid_mapping[:, 0] = mapping
    vid_mapping[:, 1] = train_data_shape[1]

    print(data_x.shape, data_y.shape)
    # The loop
    for vid_ind in range(train_data_shape[0]):
        for frame_ind, vol_num in zip(range(step, len(input_videos[vid_ind]) - step, step), range(train_data_shape[1])):
            first_frame = frame_ind - 6
            last_frame = frame_ind + 6 + 1  # adding one for slicing

            # get the new volume from the hardwired level.
            new_volume = H1(input_videos[vid_ind, first_frame:last_frame:inside_step, :, :])
            # Add the new volume to input of the first 3D-CNN layer
            data_x[vid_ind, vol_num, :, :, :] = new_volume
            # Create the label for this volume.
            # video_label carries the video label, 1 for truthful and 0 for deceptive.
            # our y_train has 2 cells, [0] deceptive and [1] for truthful
            cur_y = np.zeros((2), int)
            cur_y[video_label[vid_ind]] = 1
            # print(cur_y.shape)
            data_y[vid_ind, vol_num, :] = cur_y

    # Reshape the data to (N, dimensions)
    data_x = data_x.reshape((N, 33, 60, 40, 1))
    data_y = data_y.reshape((N, 2))

    # Micro-Expressions
    micro_expressions = get_volumes_micro_exps(micro_expressions, num_of_volumes)
    return data_x, data_y, vid_mapping, micro_expressions


def train_model(input_videos, video_label, mapping, micro_expressions,
                val_x=None, val_micro=None, val_y=None, return_best=False):
    data_x, data_y, data_mapping, data_micro = H1_preprocessing(input_videos, video_label, mapping, micro_expressions)
    model, callbacks_lst = build_model()

    if val_x is not None and val_y is not None and val_micro is not None:
        hist = model.fit([data_x, data_micro[0], data_micro[1], data_micro[2]], data_y,
                         validation_data=([val_x, val_micro[0], val_micro[1], val_micro[2]], val_y),
                         batch_size=2, epochs=20, verbose=2,
                         shuffle=True, callbacks=callbacks_lst)
        print(hist)
    else:
        hist = model.fit([data_x, data_micro[0], data_micro[1], data_micro[2]], data_y,
                         batch_size=2, epochs=20, verbose=2,
                         shuffle=True, callbacks=callbacks_lst)
        print(hist)

    if return_best:
        model.load_weights("weights.best.hdf5")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_model(model, test_videos, test_video_label, test_mapping_orig):
    """
    Call this to generate all the predictions, then
    choose based on the desired video index
    """
    test_x, test_y, test_mapping, test_micro = H1_preprocessing(test_videos, test_video_label, test_mapping_orig)
    accuracy, _, predictions = test(model, test_x, test_y, test_mapping, test_micro)
    return accuracy, predictions


def prepare_data(data_path, micro_expressions):
    """
    data_path is either
    ["/Dataset_CNN/Train" or "/Dataset_CNN/Test"]
    where each Folder contains > videos
    each video contains > clips
    """
    clips_paths, clips_mapping = get_clips_paths(data_path, ".jpg")
    clips_data = []
    clips_labels = []
    for video in tqdm(clips_paths):
        for clip in video:
            clips_data.append(read_cropped_frames(clip, (DIM_WIDTH, DIM_HEIGHT), FLAG_TO_GRAY))
            if "lie" in clip[0]:
                clips_labels.append(0)
            else:
                clips_labels.append(1)

    # Cast into Numpy Arrays
    clips_data = np.asarray(clips_data)  # input videos
    clips_labels = np.asarray(clips_labels)  # labels of the input
    clips_mapping = np.asarray(clips_mapping)  # mapping of clips to OG video
    return clips_data, clips_labels, clips_mapping, micro_expressions


def save_model(model):
    model.save("Models/Deception_Classifier_Conc.h5")
    return


def load_previous_model(model_h5_path):
    # Example: model_h5_path = "Models/Deception_Classifier_Conc.h5""
    model = load_model(model_h5_path)
    return model

