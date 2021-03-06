import os
import utils

root = "."
jpg_folder = "byclass_jpg"
path2ajpgs = os.path.join(root, jpg_folder)

# use get_videos() to import jpg images
all_vids, all_labels, catgs = utils.get_videos(path2ajpgs)

len(all_vids), len(all_labels), len(catgs)

all_vids[:3], all_labels[:3], catgs[:4]

# convert class name to integer
labels_dict = {}
ind = 0
for uc in catgs:
    labels_dict[uc] = ind
    ind += 1
print(labels_dict)

num_classes = 4
unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label] < num_classes]
unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label] < num_classes]
len(unique_ids),len(unique_labels)

# split data to training & testing dataset
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

train_ids = [unique_ids[ind] for ind in train_indx]
train_labels = [unique_labels[ind] for ind in train_indx]
print(len(train_ids), len(train_labels)) 

test_ids = [unique_ids[ind] for ind in test_indx]
test_labels = [unique_labels[ind] for ind in test_indx]
print(len(test_ids), len(test_labels))
print(test_ids)

train_ids[:5], train_labels[:5], test_ids[:5], test_labels[:5]

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
        path2imgs.sort()
        label = labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frame = self.transform(frame)
            frames.append(frame)
        if len(frames) > 0:
            frames = torch.stack(frames)
            frames = frames.permute(1,0,2,3)
        return frames, label

# Define Transform
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Create Datasets
train_ds = VideoDataset(ids=train_ids, labels=train_labels, transform=transform)
test_ds = VideoDataset(ids=test_ids, labels=test_labels, transform=transform)

imgs, label = train_ds[61]
imgs.shape, label, torch.min(imgs), torch.max(imgs)

# Create DataLoaders
from torch.utils.data import DataLoader

batch_size = 8
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=0)


# print("Printing training/testing dataloader.shape...")
# for xb, yb in train_dl:
#     print(xb.shape, yb.shape)
#     break
# for xb, yb in test_dl:
#     print(xb.shape, yb.shape)
#     break

# Define CNN Model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3,6,5)
        self.pool  = nn.MaxPool3d(2,2)
        self.conv2 = nn.Conv3d(6,16,5)
        self.conv3 = nn.Conv3d(16,16,5)
        self.gapool = nn.AdaptiveAvgPool3d((33,1,1))
        self.fc1 = nn.Linear(16*33*1*1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # Conv1: H: (240-5+2*0)/1+1 = 236, W: (320-5+2*0)/1+1 = 316, D: (151-5+2*0)/1+1 = 147
        x = self.pool(x)             # Pool1: H: (236-2+2*0)/2+1= 118,  W: (316-2+2*0)/2+1 = 158, D: (147-2+2*0)/2+1 = 73.5
        x = F.relu(self.conv2(x))    # Conv2: H: (118-5+2*0)/1+1 = 114, W: (158-5+2*0)/1+1 = 154, D: (73-5+2*0/1+1 = 69
        x = self.pool(x)             # Pool2: H: (114-2+2*0)/2+1 = 57,  W: (154-2+2*0)/2+1 = 77,  D: (69-2+2*0)/2+1 = 34.5
        x = F.relu(self.conv3(x))    # Conv3: H: (57-5+2*0)/1+1 = 53,   W: (77-5+2*0)/1+1 = 73,   D: (34-2+2*0)/1+1 = 33
        # x = self.pool(x)             # Pool3: H: (53-2+2*0)/2+1 = 26.5  W: (73-2+2*0)/2+1 = 36.5, D: (33-2+2*0)/2+1 = 16.5
        x = self.gapool(x)           # GAPool: H:1 W:1 D:33
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create CNN
net = Net()

# Define Loss Function & Optimizer
import torch.optim as optim

num_epochs = 100
current_lr = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9)

# Check GPU Availability
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Send Model to CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# Save weights
path = './models/weights_mynet_0808_2.pt'
torch.save(net.state_dict(), path)

loss_history = {
    "train": [],
    "val": [],
}
metric_history = {
    "train": [],
    "val": [],
}

# Training
from tqdm import tqdm
for epoch in range(num_epochs):
    current_lr = utils.get_lr(optimizer)
    print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

    # TRAINING
    net.train()
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(train_dl.dataset)
    for data in tqdm(train_dl):
        # calculate output and loss
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backprop and update weight
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        # add batch metric and batch loss to sum
        metric_b = utils.metrics_batch(outputs, labels)
        running_loss += loss.item()
        if metric_b is not None:
            running_metric += metric_b
    # calculate epoch loss and epoch metric
    train_loss = running_loss / float(len_data)
    train_metric = running_metric / float(len_data)
    loss_history["train"].append(train_loss)
    metric_history["train"].append(train_metric)

    # EVAULATING (TESTING)
    net.eval()
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(test_dl.dataset)
    with torch.no_grad():
        for data in tqdm(test_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            metric_b = utils.metrics_batch(outputs, labels)
            running_loss += loss.item()
            if metric_b is not None:
                running_metric += metric_b
        test_loss = running_loss / float(len_data)
        test_metric = running_metric / float(len_data)
        loss_history["val"].append(test_loss)
        metric_history["val"].append(test_metric)
    print("train loss: %.6f, test loss: %.6f, accuracy: %.2f" % (train_loss, test_loss, 100*test_metric))
    print("-"*10)
    torch.save(net.state_dict(), path)
print('Finished Training')
print("Loss Histogram = ", loss_history)
print("Metric Histogram = ", metric_history)

# net.eval()
# net.load_state_dict(torch.load(path))
# correct = 0
# total = 0
# running_loss = 0.0
# with torch.no_grad():
#     for i, data in enumerate(test_dl):
#         print( f'[Batch Number:%3d/%3d, Batch Size:%3d]' % (i+1, len(test_dl), batch_size), end="\r")
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)

#         loss = criterion(outputs, labels)
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         print(labels)
#         print(outputs.data)
#         print(predicted)
#         print('='*50)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print()
# print( f'Testing Loss:%.6f' % (running_loss/len(test_ids)) )
# print( f'Testing Accuracy: %f' % (correct/total))

# # accuracy for each class
# # count predictions for each class
# import numpy as np
# correct_pred = np.zeros((num_classes))
# total_pred = np.zeros((num_classes))

# # no gradients needed
# with torch.no_grad():
#     for i, data in enumerate(test_dl):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)
#         _, predictions = torch.max(outputs.data,1)
#         print(labels)
#         print(predictions)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[label] += 1
#             total_pred[label] += 1
#             print(correct_pred)
#             print(total_pred)
#             print('='*50)

# # print accuracy for each class
# for i, correct_num in enumerate(correct_pred):
#     accuracy = 100 * float(correct_num) / total_pred[i]
#     print("Accuracy for class {:1d} is : {:.1f} %" .format(i, accuracy))
