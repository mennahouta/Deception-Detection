# region Imports & variable declarations
# !pip install cvlib
from tqdm import tqdm
import glob
import cv2
import numpy as np
import os
from os.path import isfile, join
import cvlib
# from google.colab import files
# from google.colab.patches import cv2_imshow


# dataset_path = '/content/gdrive/My Drive/Team\'s Drive/Graduation Project/Project/Dataset'
dataset_path = "/dataset"
# endregion

# region Functions
# region I/O Related functions.


def read_videos(path):
    videos = []
    videos_fps = []  # frames per second
    videos_paths = glob.glob(path + '/*[0-9].mp4')
    videos_paths.sort()
    for video_path in videos_paths:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        videos.append(video)
        videos_fps.append(fps)
    cv2.destroyAllWindows()
    return videos, videos_fps


def read(video_class):
    if video_class == 'lie':
        videos = read_videos(dataset_path + '/Clips/Deceptive')
    else:
        videos = read_videos(dataset_path + '/Clips/Truthful')
    return videos


def get_video_frames(video):
    frames = []
    width = video.get(3)
    height = video.get(4)
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    return frames, width, height
# endregion

# region YOLO-related functions, for extracting the region of interest (person).


def yolo(frames):
    """ This function returns a list containing the bounding boxes of each frame in one video.
    If more than one bounding box are found in one frame, the bouding boxes are merged to obtain one bounding box."""
    bounding_boxes = []
    frame_no = 0
    while frame_no < len(frames):
        frame = frames[frame_no]
        frame_no += 30
        bboxes, label, confidence = cvlib.detect_common_objects(frame)
        bboxes_person = []
        for bbox, label in zip(bboxes, label):
            if label == 'person':
                bboxes_person.append(bbox)
        if len(bboxes_person) == 0:
            continue
        x_top_left = bboxes_person[0][0]
        y_top_left = bboxes_person[0][1]
        x_bottom_right = bboxes_person[0][2]
        y_bottom_right = bboxes_person[0][3]
        for i in range(1, len(bboxes_person)): # if more than one bbox, merge them
            x_top_left = min(x_top_left, bboxes_person[i][0])
            y_top_left = min(y_top_left, bboxes_person[i][1])
            x_bottom_right = max(x_bottom_right, bboxes_person[i][2])
            y_bottom_right = max(y_bottom_right, bboxes_person[i][3])
        bbox = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        bounding_boxes.append(bbox)
    return bounding_boxes  # list of bboxes, each bbox is a list representing a bbox of each (i * 20)th frame


def get_max_bounding_box(bounding_boxes):
    """ This function finds the bounding box with the largest area for one video. """
    max_bbox = []
    max_bbox_area = -1
    for bbox in bounding_boxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        bbox_area = width * height
        if bbox_area > max_bbox_area:
            max_bbox_area = bbox_area
            max_bbox = bbox
    return max_bbox


def crop_video(frames, width, height, bounding_box):
    """ This function crops a video according to the given bounding box and returns the cropped frames. """
    x_top_left = int(max(0, bounding_box[0] - 5))
    y_top_left = int(max(0, bounding_box[1] - 5))
    x_bottom_right = int(min(width - 1, bounding_box[2] + 5)) # video.get(3) = width
    y_bottom_right = int(min(height - 1, bounding_box[3] + 5)) # video.get(4) = height
    # frames = get_video_frames(video)
    cropped_frames = []
    for frame in frames:
        cropped_frame = frame[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        cropped_frames.append(cropped_frame)
        #cv2.imshow('cropped frame', cropped_frame)
        #cv2.waitKey()
    return cropped_frames
# endregion
# endregion

# region Main


# region Deceptive
deceptive_videos = read('lie')
current_directory = os.getcwd()
postyolo_dir = current_directory + dataset_path + 'PostYOLO'
os.mkdir(postyolo_dir)
deceptive_postyolo_dir = postyolo_dir + '/Deceptive'
os.mkdir(deceptive_postyolo_dir)
for i in tqdm(range(len(deceptive_videos))):
    video = deceptive_videos[i]
    frames, width, height = get_video_frames(video)
    # frames = frames[:3] # COMMENT THIS LINE AFTER TESTING
    bboxes = yolo(frames)  # list of lists(each inner list is a bbox)
    max_bbox = get_max_bounding_box(bboxes)  # list of 4 integers
    cropped_frames = crop_video(frames, width, height, max_bbox)
    video_dir = deceptive_postyolo_dir + '/trial_lie' + str(i + 1).zfill(3)
    os.mkdir(video_dir)
    for k in range(len(cropped_frames)):
        cv2.imwrite(video_dir + '/' + str(k).zfill(4) + '.jpg', cropped_frames[k])
# endregion

# region Truthful
truthful_videos = read('truth')
truthful_postyolo_dir = postyolo_dir + '/Truthful'
os.mkdir(truthful_postyolo_dir)
for i in tqdm(range(len(truthful_videos))):
    video = truthful_videos[i]
    frames, width, height = get_video_frames(video)
    # frames = frames[:3] # COMMENT THIS LINE AFTER TESTING
    bboxes = yolo(frames)  # list of lists(each inner list is a bbox)
    max_bbox = get_max_bounding_box(bboxes)  # list of 4 integers
    cropped_frames = crop_video(frames, width, height, max_bbox)
    video_dir = truthful_postyolo_dir + '/trial_truth' + str(i + 1).zfill(3)
    os.mkdir(video_dir)
    for k in range(len(cropped_frames)):
        cv2.imwrite(video_dir + '/' + str(k).zfill(4) + '.jpg', cropped_frames[k])
# endregion
# endregion
