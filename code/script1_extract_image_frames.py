# Extracts 2 frames per sec for each pornography-2k video.
import os

import utils.preproc as preproc

path2data = "./data/pornography-2k/"
sub_folder = "original"
sub_folder_jpg = "images"
path2aCatgs = os.path.join(path2data, sub_folder)




extension = ".mp4"
fps = 2
for root, dirs, files in os.walk(path2aCatgs, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        frames, vlen = preproc.get_frames(path2vid, n_frames= fps)
        frames = preproc.resize(frames)
        if 'NonPorn' in name:
            cat = 'nonporn'
        else:
            cat = 'porn'
        # ./pornography-2k/images/video-id.mp4
        path2store = path2vid.replace(sub_folder, sub_folder_jpg)
        # ./pornography-2k/images/video-id
        path2store = path2store.replace(extension, "")
        sp_path = path2store.split('/')
        #video-id
        video_id = sp_path[-1]
        # ./pornography-2k/images/
        path2store = path2store.replace(video_id,"")
        # ./pornography-2k/images/category
        path2store = os.path.join(path2store, cat)
        
        print(path2store)
        os.makedirs(path2store, exist_ok= True)
        preproc.store_frames(frames, path2store, video_id)
    print("-"*50)    
