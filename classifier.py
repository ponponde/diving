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
labels_dict

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
        print(path2imgs)
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

batch_size = 10
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=0)
for xb, yb in train_dl:
    print(xb.shape, yb.shape)
    break
for xb, yb in test_dl:
    print(xb.shape, yb.shape)
    break

# Define CNN Model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3,6,5)
        self.pool = nn.MaxPool3d(2,2)
        self.conv2 = nn.Conv3d(6,16,5)
        self.conv3 = nn.Conv3d(16,16,5)
        self.fc1 = nn.Linear(16*151*26*36, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # Conv1: H: (240-5+2*0)/1+1 = 236, W: (320-5+2*0)/1+1 = 316
        x = self.pool(x)             # Pool1: H: (236-2+2*0)/2+1= 118, W: (316-2+2*0)/2+1 = 158
        x = F.relu(self.conv2(x))    # Conv2: H: (118-5+2*0)/1+1 = 114, W: (158-5+2*0)/1+1 = 154
        x = self.pool(x)             # Pool2: H: (114-2+2*0)/2+1 = 57, W: (154-2+2*0)/2+1 = 77
        x = F.relu(self.conv3(x))    # Conv3: H: (57-5+2*0)/1+1 = 53, W: (77-5+2*0)/1+1 = 73
        x = self.pool(x)             # Pool3: H: (53-2+2*0)/2+1 = 26.5 W: (73-2+2*0)/2+1 = 36.5
        print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create CNN
net = Net()

# Define Loss Function & Optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Check GPU Availability
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Send Model to CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# Training
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

