import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def get_frames(filename, n_frames= 1):
    """Extracts the frames of a video

    Args:
        filename (str): the path to the video
        n_frames (int, optional): number of frames per sec to extract. Defaults to 1.

    Returns:
        frames: the extracted frames
        v_len: the video length
    """
    frames = []
    v_cap = cv2.VideoCapture(filename)
    fps = v_cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = v_len/fps
    n_frames = int(n_frames*duration)
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def resize(frames):
    """Resizes the given frames

    Args:
        frames (_type_): The frames to resize

    Returns:
        _type_: the resized frames
    """
    h, w = 224, 224
    frames_tr = []
    for frame in frames:
        frame_tr = cv2.resize(frame, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        frames_tr.append(frame_tr)
    return frames_tr


def store_frames(frames, path2store, video_id):
    """Stores the given video frames to a given directory.

    Args:
        frames (_type_): the frames
        path2store (str): the directory to store the frames 
        video_id (str): the video's id
    """    """
    """
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, video_id+"fr"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)
