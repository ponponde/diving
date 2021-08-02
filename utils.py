import os
import cv2
import numpy as np

# Read a video and convert to a list
def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len

# Store frames from a video as jpg images
def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path2img = os.path.join(path2store, "frame" + str(ii).zfill(3) + ".jpg")
        cv2.imwrite(path2img, frame)
        
# Get frames from jpg folder
def get_videos(path2ajpgs):                # ./jpg_folder
    listOfCats = os.listdir(path2ajpgs)    # ./jpg_folder/60 & ./jpg_folder/70 ...
    listOfCats.sort()
    ids = []
    labels = []
    for catg in listOfCats:                # ./jpg_folder/60/020 & ./jpg_folder/60/047 ...
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)    # 020/frame0.jpg & 020/frame1.jpg ...
        path2SubCats = [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2SubCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats
