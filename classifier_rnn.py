import os
import utils

root = "."
jpg_folder = "byclass_jpg"
path2ajpgs = os.path.join(root, jpg_folder)

# use get_videos() to import jpg images
all_vids, all_labels, catgs = utils.get_videos(path2ajpgs)

print("Check if get_videos run correctly...")
print("all_vids:", all_vids[:5])
print("all_labels:", all_labels[:5])
print("catgs:", catgs)
print("length of all_vids, all_labels, catgs:", len(all_vids), len(all_labels), len(catgs))
print("="*50)

# convert class name to integer
labels_dict = {}
for i, label in enumerate(catgs):
    labels_dict[label] = i 
print("labels_dict:", labels_dict)
print("="*50)

num_classes = len(catgs)

# select only data with label in labels_dict
unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label] < num_classes]
unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label] < num_classes]

# split data to training & testing dataset
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
train_indx, test_indx = next(sss.split(unique_ids, unique_labels))
train_ids = [unique_ids[ind] for ind in train_indx]
train_labels = [unique_labels[ind] for ind in train_indx]
test_ids = [unique_ids[ind] for ind in test_indx]
test_labels = [unique_labels[ind] for ind in test_indx]
print(train_ids[:5], train_labels[:5], test_ids[:5], test_labels[:5])
print("len(train_ids): ", len(train_ids), "len(train_labels): ", len(train_labels)) 
print("len(test_ids): ", len(test_ids), "len(test_labels): ", len(test_labels))
print("test_labels: ", test_labels)
print("="*50)

# Define Dataset
from torch.utils.data import Dataset
import glob
from PIL import Image
import torch

class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform):
        self.ids = ids
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        # Get jpgs generated from video "ids[idx]"
        path2imgs = glob.glob(self.ids[idx] + "/*.jpg")
        label = labels_dict[self.labels[idx]]
        frames = []
        path2imgs.sort()
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frame = self.transform(frame)
            frames.append(frame)
        if len(frames) > 0:
            frames = torch.stack(frames)
            # frames = frames.permute(1,0,2,3)
        return frames, label

# Define Transform Parameters
model_type = "rnn"
timesteps = 16
if model_type == "rnn":
    h, w = 224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    h, w = 112, 112
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

#Define Transform
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((h,w)),
    # transforms.RandomHorizontalFlip(p=0.5),   # flip images randomly with a given probability
    # transforms.RandomAffine(degree=0, translate=(0.1,0.1)),   # keep image center invariant
    transforms.ToTensor(),
    transforms.Normalize(mean, std),    # z = (x - mean) / std
])

# Create Training/Testing Datasets
train_ds = VideoDataset(ids=train_ids, labels=train_labels, transform=transform)
imgs, label = train_ds[10]
print("Training dataset:")
print("img.shape: ", imgs.shape, "label: ", label) 
print("torch.min(imgs): ", torch.min(imgs), "torch.max(imgs): ", torch.max(imgs))

test_ds = VideoDataset(ids=test_ids, labels=test_labels, transform=transform)
imgs, label = test_ds[1]
print("Testing dataset:")
print("img.shape: ", imgs.shape, "label: ", label) 
print("torch.min(imgs): ", torch.min(imgs), "torch.max(imgs): ", torch.max(imgs))
print("="*50)


# Define DataLoaders
from torch.utils.data import DataLoader
def collate_fn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    # If using r3d_18, transpose frame with channel
    # imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor

# Create DataLoaders
batch_size = 2
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn)

print("Printing training/testing dataloader.shape...")
for xb, yb in train_dl:
    print(xb.shape, yb.shape)
    break
for xb, yb in test_dl:
    print(xb.shape, yb.shape)
    break
print("="*50)

# Define CNN Model
import torch.nn as nn
class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()

        num_classes     = params_model["num_classes"]
        dr_rate         = params_model["dr_rate"]
        pretrained      = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers  = params_model["rnn_num_layers"]

        baseModel    = models.resnet18(pretrained = pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()   # make original fully connect layer to empty layer

        # main resnet18 framework
        self.baseModel = baseModel
        # randomly zero some of the elements of the input tensor with probability p
        self.dropout   = nn.Dropout(dr_rate)
        self.rnn       = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1       = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        batch_size, frames, c, h, w = x.shape
        ii = 0
        # first, train with resnet18
        y = self.baseModel((x[:,ii]))
        # second, train with rnn(lstm)
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, frames):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        # drop out some elements
        out = self.dropout(out[:,-1])
        # last fully connect linear layer
        out = self.fc1(out)
        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# Create Model
from torchvision import models
from torch import nn

if model_type == "rnn":
    params_model={
        "num_classes"    : num_classes,
        "dr_rate"        : 0.1,
        "pretrained"     : True,
        "rnn_num_layers" : 1,
        "rnn_hidden_size": 100,
    }
    model = Resnt18Rnn(params_model)

with torch.no_grad():
    if model_type == "rnn":
        x = torch.zeros(1, 16, 3, h, w)
    y = model(x)
    print("Print shape of our model output", y.shape)
    print("="*50)

# Send Model to CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Save model
path2weights = "./models/weights_rnn_0809_1.pt"
os.makedirs("./models", exist_ok=True)
# torch.save(model.state_dict(), path2weights)

# Training
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

loss = nn.CrossEntropyLoss(reduction="sum")
opt  = optim.SGD(model.parameters(), lr=1e-2)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=1)

params_train={
    "num_epochs"  : 50,
    "optimizer"   : opt,
    "loss_func"   : loss,
    "train_dl"    : train_dl,
    "val_dl"      : test_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    # "path2weights": "./models/weights_" + model_type + ".pt",
    "path2weights": path2weights,
}

import copy
def train_val(model, params):
    num_epochs   = params["num_epochs"]
    loss_func    = params["loss_func"]
    opt          = params["optimizer"]
    train_dl     = params["train_dl"]
    val_dl       = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {
        "train": [],
        "val": [],
    }
    metric_history = {
        "train": [],
        "val": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = utils.get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = utils.loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # set module in evaluation mode, deactivate dropout layer
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = utils.loss_epoch(model, loss_func, val_dl, sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # update(reduce) learning rate when metric stop improving
        lr_scheduler.step(val_loss)
        if current_lr != utils.get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" % (train_loss, val_loss, 100*val_metric))
        print("-"*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


model, loss_hist, metric_hist = train_val(model, params_train)
print("Loss Histogram: ", loss_hist)
print("Metric_hist: ", metric_hist)

