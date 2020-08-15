import glob
import os
import numpy as np

current_dir = os.getcwd()
dataset_path = current_dir + '/Dataset_VGG'
train_path = dataset_path + '/Train'
test_path = '/content/gdrive/My Drive/Team\'s Drive/Graduation Project/Dataset_VGG/test'
labels_path = dataset_path + '/initial_labels'

# from google.colab import drive
# drive.mount('/content/gdrive')


def get_inner_paths(path):
    inner_paths = glob.glob(path + '/*')
    inner_paths.sort()
    inner_names = [os.path.basename(inner_path) for inner_path in inner_paths]
    return inner_paths, inner_names


truthful_labels = np.load(labels_path + '/truthful_labels.npy')
deceptive_labels = np.load(labels_path + '/deceptive_labels.npy')


def get_new_labels(path):
    _, video_names = get_inner_paths(path)
    labels = []
    # video_name examples: trial_lie030 | trial_truth007
    for video_name in video_names:
        if video_name[6] == 't':  # truthful video
            video_num = int(video_name[11:]) - 1
            labels.append(truthful_labels[video_num])
        else:
            video_num = int(video_name[9:]) - 1
            labels.append(deceptive_labels[video_num])
    return labels


train_labels = get_new_labels(train_path)
np.save(dataset_path + '/train_labels.npy', np.asarray(train_labels))

test_labels = get_new_labels(test_path)
np.save(dataset_path + '/test_labels.npy', np.asarray(test_labels))