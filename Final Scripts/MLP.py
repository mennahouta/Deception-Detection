# region Imports
import numpy as np
import cv2
import glob

from tqdm import tqdm

from keras.layers import (Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.models import Sequential
from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
# endregion

# region Constants
IMG_DIM = 214
FLAG_TO_GRAY = False
# endregion


# region Helper Functions
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
def MLP():
    """
    Complete Model Architecture
    """
    classifier = Sequential(
        [
            Conv3D(32, activation='relu', kernel_size=(5, 5, 5), padding='valid',
                   input_shape=(28, IMG_DIM, IMG_DIM, 3), data_format='channels_last', name="CONV"),  # 112
            MaxPooling3D(pool_size=(3, 3, 3), name="POOL"),
            Flatten(),
            Dense(units=300, activation='relu', name="Vf"),
            Dense(units=1024, activation='relu', name='Hidden_Layer'),
            # Conv3D(300, activation='relu', kernel_size=(36, 70, 70), padding='valid',
            #                                               data_format='channels_last', name="CONV_FC1"),
            # Conv3D(1024, activation='relu', kernel_size=(1, 1, 1), padding='valid',
            #                                               data_format='channels_last', name="CONV_FC2"),
            Dropout(.5),
            # Flatten(),
            Dense(units=2, activation='softmax', name='MLP_output'),
        ], name="Visual_MLP"
    )
    return classifier


def test(model, x_test, y_test, clips_mapping):
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
        y_pred = model.predict(x_test[prev_batch_index:(prev_batch_index + num_cells)])
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
    model = MLP()
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    file_path = "weights.best.hdf5"
    mcp_save = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4,
                                       mode='min')
    callbacks_lst = [mcp_save]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model, callbacks_lst


def MLP__preprocessing(input_videos, video_label, mapping):
    # From 113 frames down-sample to reach 28
    input_videos = input_videos[:, ::4, :, :, :]
    input_videos = input_videos[:, :28, :, :, :]
    # Make the labels one-hot-encoded
    video_label = to_one_hot(video_label)
    # No sub-clipping, hence the volume of the mapping is 1
    vid_mapping = np.empty((mapping.shape[0], 2), dtype='int32')
    vid_mapping[:, 0] = mapping
    vid_mapping[:, 1] = 1
    return input_videos, video_label, vid_mapping


def train_model(input_videos, video_label, mapping, val_x=None, val_y=None, return_best=False):
    data_x, data_y, data_mapping = MLP__preprocessing(input_videos, video_label, mapping)
    model, callbacks_lst = build_model()

    if val_x is not None and val_y is not None:
        hist = model.fit(data_x, data_y, validation_data=(val_x, val_y), batch_size=2, epochs=20, verbose=2,
                         shuffle=True, callbacks=callbacks_lst)
        print(hist)
    else:
        hist = model.fit(data_x, data_y, batch_size=2, epochs=20, verbose=2,
                         shuffle=True, callbacks=callbacks_lst)
        print(hist)

    if return_best:
        model.load_weights("weights.best.hdf5")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_model(model, test_videos, test_video_label, test_mapping_orig):
    test_x, test_y, test_mapping = MLP__preprocessing(test_videos, test_video_label, test_mapping_orig)
    accuracy, predictions, _ = test(model, test_x, test_y, test_mapping)
    return accuracy, predictions


def prepare_data(data_path):
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
            clips_data.append(read_cropped_frames(clip, (IMG_DIM, IMG_DIM), FLAG_TO_GRAY))
            if "lie" in clip[0]:
                clips_labels.append(0)
            else:
                clips_labels.append(1)

    # Cast into Numpy Arrays
    clips_data = np.asarray(clips_data)  # input videos
    clips_labels = np.asarray(clips_labels)  # labels of the input
    clips_mapping = np.asarray(clips_mapping)  # mapping of clips to OG video
    return clips_data, clips_labels, clips_mapping


def save_model(model):
    model.save("Models/Deception_Classifier_MLP.h5")
    return


def load_previous_model(model_h5_path):
    # Example: model_h5_path = "Models/Deception_Classifier_MLP.h5""
    model = load_model(model_h5_path)
    return model

