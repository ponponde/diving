import os
import utils

root = "."
video_folder = "byclass"
jpg_folder = "byclass_jpg"
path2aCatgs = os.path.join(root, video_folder)

listOfCategories = os.listdir(path2aCatgs)
listOfCategories, len(listOfCategories)

# use get_frames() & store_frames() to convert videos to jpg images
extension = ".avi"
n_frames = 151

for root, dirs, files in os.walk(path2aCatgs, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        frames, v_len = utils.get_frames(path2vid, n_frames=n_frames)
        path2store = path2vid.replace(video_folder, jpg_folder)
        path2store = path2store.replace(extension, "")
        # print(path2store)
        os.makedirs(path2store, exist_ok=True)
        utils.store_frames(frames, path2store)
    print("Successfuly convert videos in " + root + " to images")
    # print("-"*50)