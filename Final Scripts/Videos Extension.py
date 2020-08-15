# Imports & variable declarations
from tqdm import tqdm
import glob
import cv2
import numpy as np
import os
from os.path import isfile, join

#from google.colab import drive
#drive.mount('/content/gdrive')

maximum_frames = 1440
current_dir = os.getcwd()
train_path = current_dir + '/Dataset(Train-Test)/train'
test_path = current_dir + '/Dataset(Train-Test)/test'

# I/O related functions.
def get_inner_paths(path):
    inner_paths = glob.glob(path + '/*')
    inner_paths.sort()
    inner_names = [os.path.basename(inner_path) for inner_path in inner_paths]
    return inner_paths, inner_names

def get_frames(video_path):
    frame_paths, _ = get_inner_paths(video_path)
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    return frames


# Video extension with noise related functions.
def add_frames_with_noise(frames):
    print('\nNumber of frames: ', len(frames))
    if len(frames) < maximum_frames:
        new_frames = extend_video_with_noise(frames)
    elif len(frames) > maximum_frames:
        new_frames = shrink_video(frames)
    else:
        new_frames = frames
    print("New number of frames: ", len(new_frames))
    return new_frames


def shrink_video(frames):
    extra_frames_no = len(frames) - maximum_frames 
    
    if extra_frames_no < 1:
        return frames
    
    skipping_rate = maximum_frames // extra_frames_no + 1
    
    shrunken_video_frames = []
    skipped_frames = []
    #print('In shrink_video')
    for frame_index in range(len(frames)):
        frame = frames[frame_index]
        if len(shrunken_video_frames) == maximum_frames:
            break
        if frame_index % skipping_rate == 0:
            skipped_frames.append(frame)
            continue
        shrunken_video_frames.append(frame)
    index = skipping_rate
    while len(shrunken_video_frames) < 1440:
        shrunken_video_frames.insert(index, frames[index])
        index += skipping_rate
    return shrunken_video_frames


def extend_video_with_noise(frames):
    original_frames = frames

    if len(original_frames) == 0:
        return frames

    frames_with_noise = add_noise_to_frames(original_frames)

    final_frames = original_frames
    repeat_frames = maximum_frames // len(original_frames)
    i = 0
    while i in range(repeat_frames - 1):
        final_frames.extend(frames_with_noise)
        i += 1
    i = 0
    if len(final_frames) < maximum_frames:
        while len(final_frames) < maximum_frames and i < len(frames_with_noise):
            final_frames.append(frames_with_noise[i])
            i += 1
    return final_frames


def add_noise_to_frames(frames):
    noisy_frames = []
    for frame in frames:
        noisy_frames.append(add_noise_to_frame(frame))
    return noisy_frames


def add_noise_to_frame(frame):
    mean = 0
    var = 20
    sigma = var ** 0.5
    #frame = resize_frame(frame)
    height, width, _ = frame.shape
    gaussian = np.random.normal(mean, sigma, (height, width)) #  np.zeros((224, 224), np.float32)
    noisy_image = np.zeros(frame.shape, np.float32)

    if len(frame.shape) == 2:
        noisy_image = frame + gaussian
    else:
        noisy_image[:, :, 0] = frame[:, :, 0] + gaussian
        noisy_image[:, :, 1] = frame[:, :, 1] + gaussian
        noisy_image[:, :, 2] = frame[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_frame = noisy_image.astype(np.uint8)
    
    #cv2.imshow("img", frame)
    #cv2.imshow("noise", noisy_frame)
    #cv2.waitKey(0)

    return noisy_frame


def save_frames(path, frames):
    print(path)
    os.mkdir(path)
    for frame_index in range(len(frames)):
        frame_path = path + '/' + str(frame_index).zfill(4) + '.jpg'
        cv2.imwrite(frame_path, frames[frame_index])
    return


# MAIN
def main(path, save_path):
    video_paths, video_names = get_inner_paths(path)
    for video_path, video_name in zip(video_paths, video_names):
        frames = get_frames(video_path)
        new_frames = add_frames_with_noise(frames)
        save_frames(save_path + '/' + video_name, new_frames)
    return

vgg19_dir = current_dir + "/Dataset_VGG"
os.mkdir(vgg19_dir)
train_vgg19_dir = vgg19_dir + "/Train"
os.mkdir(train_vgg19_dir)
main(train_path, train_vgg19_dir)

test_vgg19_dir = vgg19_dir + "/Test"
os.mkdir(test_vgg19_dir)
main(test_path, test_vgg19_dir)


# Validation
def validate_number_of_frames(path):
    videos_paths, _ = get_inner_paths(path)
    error_paths = []
    for video_path in videos_paths:
        frames, _ = get_inner_paths(video_path)
        if len(frames) != maximum_frames:
            # print(video_path, ' ', len(frames))
            error_paths.append(video_path)
    return len(videos_paths), error_paths


number_of_train_vids, error_paths = validate_number_of_frames(train_vgg19_dir)
print('Number of train videos: ', number_of_train_vids)
print('Number of train videos with wrong number of frames: ', len(error_paths))
if len(error_paths) != 0:
    print('Error paths: ', error_paths)

number_of_test_vids, error_paths = validate_number_of_frames(test_vgg19_dir)
print('Number of test videos: ', number_of_test_vids)
print('Number of test videos with wrong number of frames: ', len(error_paths))
if len(error_paths) != 0:
    print('Error paths: ', error_paths)

