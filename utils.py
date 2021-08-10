import os
import cv2
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook

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
        print(path2img)
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


# Get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

import torch
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output, target)
    # update gradient 
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    # for each batch of dataloader
    # for xb, yb in tqdm_notebook(dataset_dl):
    for xb, yb in tqdm(dataset_dl):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        # calculate loss for one batch
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check is True:
            break
    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric