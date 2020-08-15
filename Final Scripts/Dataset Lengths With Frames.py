import subprocess
import glob
import cv2
import numpy as np
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from statistics import mean 

#Reference for outliers code: https://github.com/arshren/MachineLearning/blob/master/Identifying%20outliers.ipynb


def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    frames = int(clip.fps * clip.duration)
    return (clip.duration, frames)


def Get_Videos_Length(path):
    videos_paths = glob.glob(path + '/*[0-9].mp4')
    videos_lengths = []
    for video_path in videos_paths:
        video_length, frames = get_video_length(video_path)
        videos_lengths.append((video_length, frames))
    return videos_lengths


#MAIN
current_dir = os.getcwd()
dataset_path = current_dir + "/Dataset/Clips"
truthful_videos_length = Get_Videos_Length(dataset_path + "/Truthful")
deceptive_videos_length = Get_Videos_Length(dataset_path + "/Deceptive")
total_videos_length = []
total_videos_length.extend(truthful_videos_length)
total_videos_length.extend(deceptive_videos_length)
print(len(truthful_videos_length))
print(len(deceptive_videos_length))


total_videos_length.sort(key=lambda x:x[1])
average = []
for video in total_videos_length:
    average.append(video[1])
print(mean(average))


total_videos_length.sort(key=lambda x:x[1])
print(total_videos_length)

"""outliers = []
def detect_outlier(data_1):
    
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
       
    for y in data_1:
        z_score = (y - mean_1)/std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = detect_outlier(total_videos_length)
print(outlier_datapoints)"""