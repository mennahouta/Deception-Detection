# region Imports and Constants
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)

current_dir = os.getcwd()
train_path = current_dir + "/Dataset_CNN/train"
test_path = current_dir + "/Dataset_CNN/test"
cnn_numpys_path = current_dir + "/Dataset_CNN"
micro_exp_path =  current_dir + "/Dataset/Annotation/Microexpressions.csv"
# endregion

# region Functions' Definitions
def get_inner_paths(path):
    inner_paths = glob.glob(path + '/*')
    inner_paths.sort()
    inner_names = [os.path.basename(inner_path) for inner_path in inner_paths]
    return inner_paths, inner_names

def get_subs_micros(video_paths, video_names, micro_exp_np):
    subclips_micro_exp = []
    for video_path, video_name in zip(video_paths, video_names):
        _, subclips_names = get_inner_paths(video_path)
        index = int(video_name[-3:]) - 1
        if video_name[6] == 't':  # trial_truth
            index += 61
        for i in range(len(subclips_names)):
            subclips_micro_exp.append(micro_exp_np[index][1:40])
    return np.asarray(subclips_micro_exp)
# endregion

# region Main
train_paths, train_names = get_inner_paths(train_path)
test_paths, test_names = get_inner_paths(test_path)
micro_exp_np = pd.read_csv(micro_exp_path).to_numpy()

# to get the micro-expressions of video with index 0:
print('Micro Expressions of video 0:\n', micro_exp_np[0][1:40])
print('Length: ', len(micro_exp_np[0][1:40]))
print('Type: ', type(micro_exp_np[0][1:40]))

train_subclips_micro_exp_np = get_subs_micros(train_paths, train_names, micro_exp_np)
test_subclips_micro_exp_np = get_subs_micros(test_paths, test_names, micro_exp_np)

np.save(cnn_numpys_path + '/train_micro_exps.npy', train_subclips_micro_exp_np)
np.save(cnn_numpys_path + '/test_micro_exps.npy', test_subclips_micro_exp_np)
# endregion