# Dividing each video into equal-sized subclips (same number of frames).

# region Imports and constants
import cv2
import glob
import os
from tqdm import tqdm

# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)

min_frames = 113
current_dir = os.getcwd()
train_path = current_dir + '/Dataset(Train-Test)/Train'
test_path = current_dir + '/Dataset(Train-Test)/Test'


# endregion

# region Functions
def get_inner_paths(path):
    inner_paths = glob.glob(path + '/*')
    inner_paths.sort()
    inner_names = [os.path.basename(inner_path) for inner_path in inner_paths]
    return inner_paths, inner_names


def get_frames(video_path):
    frame_paths, _ = get_inner_paths(video_path)
    print(frame_paths)
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    return frames


def find_min_frames():
    min_frames = 100000
    train_paths, _ = get_inner_paths(train_path)
    test_paths, _ = get_inner_paths(test_path)
    for path in train_paths + test_paths:
        frames = get_frames(path)
        if len(frames) < min_frames:
            min_frames = len(frames)
    return min_frames


def divide_save_subclips(video_path, video_name, subclips_path):
    frames = get_frames(video_path)
    folder_name = subclips_path + '/' + video_name
    print(folder_name)
    os.mkdir(folder_name)
    subvideos = len(frames) // min_frames
    print('frames: ', len(frames), 'subvideos: ', subvideos)
    for subvideo_num in range(subvideos):
        print(subvideo_num)
        subvideo_name = str(subvideo_num).zfill(3)
        subvideo_dir = folder_name + '/' + subvideo_name
        os.mkdir(subvideo_dir)
        for j in range(min_frames):
            frame_index = j + subvideo_num * min_frames
            frame_path = subvideo_dir + '/' + str(frame_index).zfill(4) + '.jpg'
            cv2.imwrite(frame_path, frames[frame_index])
    remaining_frames = len(frames) % min_frames
    subvideo_num = subvideos
    if remaining_frames > 0.5 * min_frames:
        subvideo_name = str(subvideo_num).zfill(3)
        subvideo_dir = folder_name + '/' + subvideo_name
        os.mkdir(subvideo_dir)
        missing = min_frames - remaining_frames
        start_index = len(frames) - remaining_frames - missing
        for frame_index in range(start_index, len(frames)):
            frame_path = subvideo_dir + '/' + str(frame_index).zfill(4) + '.jpg'
            cv2.imwrite(frame_path, frames[frame_index])
    return


# endregion

# region Main


# min_frames = find_min_frames()
# print(min_frames)

cnn_dir = current_dir + '/Dataset_CNN'
os.mkdir(cnn_dir)
train_cnn_dir = cnn_dir + '/Train'
os.mkdir(train_cnn_dir)
train_paths, train_names = get_inner_paths(train_path)
for path, name in tqdm(zip(train_paths, train_names)):
    divide_save_subclips(path, name, train_cnn_dir)

test_cnn_dir = cnn_dir + '/Test'
os.mkdir(test_cnn_dir)
test_paths, test_names = get_inner_paths(test_path)
for path, name in tqdm(zip(test_paths, test_names)):
    divide_save_subclips(path, name, test_cnn_dir)
# endregion

# region Validation
cnn_train_paths, _ = get_inner_paths(train_cnn_dir)
cnn_test_paths, _ = get_inner_paths(test_cnn_dir)
print('Number of train videos: ', len(cnn_train_paths))
print('Number of test videos: ', len(cnn_test_paths))

mis_split_paths = []
for path in cnn_train_paths + cnn_test_paths:
    subpaths, _ = get_inner_paths(path)
    for subpath in subpaths:
        frames_paths, _ = get_inner_paths(subpath)
        if len(frames_paths) != 113:
            mis_split_paths.append(subpath)
print('Number of mis-split videos: ', len(mis_split_paths))
if len(mis_split_paths) != 0:
    print('Mis-split paths: ', mis_split_paths)


def validate_number_of_frames(paths, split_paths):
    error_paths = []
    print(paths)
    print(split_paths)
    for path, split_path in zip(paths, split_paths):
        _, frame_numbers = get_inner_paths(path)
        subpaths, _ = get_inner_paths(split_path)
        last_subpath = subpaths[-1]
        _, last_frames_numbers = get_inner_paths(last_subpath)
        last_frame_split = int(last_frames_numbers[-1][:4])
        last_frame = int(frame_numbers[-1][:4])
        if last_frame_split != last_frame and last_frame - last_frame_split > 0.5 * min_frames:
            error_paths.append(split_path)
    return error_paths


train_paths, _ = get_inner_paths(train_path)
cnn_train_paths, _ = get_inner_paths(train_cnn_dir)
error_paths = validate_number_of_frames(train_paths, cnn_train_paths)
print('Number of train videos with missing frames: ', len(error_paths))
if len(error_paths) != 0:
    print('Error paths: ', error_paths)

test_paths, _ = get_inner_paths(test_path)
cnn_test_paths, _ = get_inner_paths(test_cnn_dir)
error_paths = validate_number_of_frames(test_paths, cnn_test_paths)
print('Number of test videos with missing frames: ', len(error_paths))
if len(error_paths) != 0:
    print('Error paths: ', error_paths)
# endregion
