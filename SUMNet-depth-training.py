#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models
from torchvision import transforms 
from torchvision.transforms import CenterCrop
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import glob
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


BATCH_SIZE = 16
gap_save = 5
out = 'sumnet_model/'
if not os.path.exists(out):
    os.makedirs(out)
num_epochs = 50


# In[ ]:

# Preparing the dataset for model processing
class SimColDataset(Dataset):
    def __init__(self,filenames,transform=True,mean_f=(0.485, 0.456, 0.406),
                 std_f=(0.229, 0.224, 0.225),dim=[448,448]):
        
        self.filenames = filenames
        self.dim = dim
        
        self.transform_depth = transforms.Compose([
                      transforms.ToPILImage(),  
                      transforms.Resize(448),
                      transforms.ToTensor() 
                ])
        
        if not transform:
            self.transform_frame = transforms.Compose([   
                      transforms.ToPILImage(),
                      transforms.Resize(448),
                      transforms.ToTensor()
                ])
        else:
            self.transform_frame = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize(448),
                      transforms.ToTensor(),
                      transforms.Normalize(mean_f,std_f)       
                ])

        self.length = len(self.filenames)

    def __len__(self):
            return self.length
        
    def __getitem__(self, idx):
            frame_path = self.filenames[idx]
            frame = None
            
            path_prefix = os.path.dirname(frame_path)
            frame_name = os.path.basename(frame_path)
            depth_name = 'Depth_'+frame_name.split('_')[-1]
            depth_path = os.path.join(path_prefix,depth_name)
            depth = None
            
            if os.path.isfile(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)
                
            if os.path.isfile(depth_path):
                depth = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
                depth = depth.astype(np.uint8)
                
            return self.transform_frame(frame),self.transform_depth(depth) 


# In[ ]:


def get_mean_and_std_frame(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean, std


# In[ ]:

# Save the paths of the training data folders in train_file.txt
train_file = 'train_file.txt'
with open(train_file) as f:
    trn_folders = f.read().splitlines() 
    
train_filenames = []
for class_path in trn_folders:
        imgs_frame = sorted(glob.glob(class_path + "FrameBuffer_*.png"))
        train_filenames.extend(imgs_frame)


# In[ ]:

# Save the paths of the validation data folders in val_file.txt
val_file = 'val_file.txt'
with open(val_file) as f:
    val_folders = f.read().splitlines() 
    
val_filenames = []
for class_path in val_folders:
        imgs_frame = sorted(glob.glob(class_path + "FrameBuffer_*.png"))
        val_filenames.extend(imgs_frame)


# In[ ]:

# Save the paths of the test data folders in test_file.txt
test_file = 'test_file.txt'
with open(test_file) as f:
    test_folders = f.read().splitlines() 
    
test_filenames = []
for class_path in test_folders:
        imgs_frame = sorted(glob.glob(class_path + "FrameBuffer_*.png"))
        test_filenames.extend(imgs_frame)



# In[ ]:


data_initial = SimColDataset(train_filenames,False)
loader = DataLoader(data_initial,
          batch_size=BATCH_SIZE,
          shuffle=True,
          num_workers=8,
          drop_last=True)

mean_f, std_f = get_mean_and_std_frame(loader)


# In[ ]:

# Using wandb to log the training weights and losses
import wandb
wandb.login()
wandb.init(project="sumnet")
wandb.log({"mean_f": str(mean_f), "std_f": str(std_f)})


# In[ ]:

train_data = SimColDataset(train_filenames,True,mean_f,std_f)
trainloader = DataLoader(train_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True)
print("Training data loaded...")


# In[ ]:


val_data = SimColDataset(val_filenames,True,mean_f,std_f)
valloader = DataLoader(val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True)
print("Validation data loaded...")


# In[ ]:

# Model definition
class conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SUMNet(nn.Module):
    def __init__(self):
        super(SUMNet, self).__init__()
        
        self.encoder   = models.vgg11(pretrained = True).features
        self.conv1     = nn.Sequential(
                            self.encoder[0],
                            nn.BatchNorm2d(64)
                        )
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv2     = nn.Sequential(
                            self.encoder[3],
                            nn.BatchNorm2d(128)
                        )
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = nn.Sequential(
                            self.encoder[6],
                            nn.BatchNorm2d(256)
                        )
        self.conv3b    = nn.Sequential(
                            self.encoder[8],
                            nn.BatchNorm2d(256)
                        )
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = nn.Sequential(
                            self.encoder[11],
                            nn.BatchNorm2d(512)
                        )
        self.conv4b    = nn.Sequential(
                            self.encoder[13],
                            nn.BatchNorm2d(512)
                        )
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = nn.Sequential(
                            self.encoder[16],
                            nn.BatchNorm2d(512)
                        )
        self.conv5b    = nn.Sequential(
                            self.encoder[18],
                            nn.BatchNorm2d(512)
                        )
        self.pool5     = nn.MaxPool2d(2, 2, return_indices = True)
        
        self.unpool5   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = conv_bn(1024, 512)
        self.donv5a    = conv_bn(512, 512)
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = conv_bn(1024, 512)
        self.donv4a    = conv_bn(512, 256)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = conv_bn(512, 256)
        self.donv3a    = conv_bn(256,128)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv2     = conv_bn(256, 64)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv1     = conv_bn(128, 32)
        self.output    = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        
        conv1          = F.relu(self.conv1(x), inplace = True)
        pool1, idxs1   = self.pool1(conv1)
        conv2          = F.relu(self.conv2(pool1), inplace = True)
        pool2, idxs2   = self.pool2(conv2)
        conv3a         = F.relu(self.conv3a(pool2), inplace = True)
        conv3b         = F.relu(self.conv3b(conv3a), inplace = True)
        pool3, idxs3   = self.pool3(conv3b)
        conv4a         = F.relu(self.conv4a(pool3), inplace = True)
        conv4b         = F.relu(self.conv4b(conv4a), inplace = True)
        pool4, idxs4   = self.pool4(conv4b)
        conv5a         = F.relu(self.conv5a(pool4), inplace = True)
        conv5b         = F.relu(self.conv5b(conv5a), inplace = True)
        pool5, idxs5   = self.pool5(conv5b)
        
        unpool5        = torch.cat([self.unpool5(pool5, idxs5), conv5b], 1)
        donv5b         = F.relu(self.donv5b(unpool5), inplace = True)
        donv5a         = F.relu(self.donv5a(donv5b), inplace = True)
        unpool4        = torch.cat([self.unpool4(donv5a, idxs4), conv4b], 1)
        donv4b         = F.relu(self.donv4b(unpool4), inplace = True)
        donv4a         = F.relu(self.donv4a(donv4b), inplace = True)
        unpool3        = torch.cat([self.unpool3(donv4a, idxs3), conv3b], 1)
        donv3b         = F.relu(self.donv3b(unpool3), inplace = True)
        donv3a         = F.relu(self.donv3a(donv3b))
        unpool2        = torch.cat([self.unpool2(donv3a, idxs2), conv2], 1)
        donv2          = F.relu(self.donv2(unpool2), inplace = True)
        unpool1        = torch.cat([self.unpool1(donv2, idxs1), conv1], 1)
        donv1          = F.relu(self.donv1(unpool1), inplace = True)
        output         = self.output(donv1)
        # return F.softmax(output,dim=1)
        return torch.sigmoid(output)


# In[ ]:


eps = 1e-8
class ScaleInvariantL2(nn.Module):
    def __init__(self, lamda = 0.5):
        super(ScaleInvariantL2, self).__init__()
        self.lamda = lamda
 
    def forward(self, pred, gt):

        pred = pred.view(-1)
        gt = gt.view(-1)

        d = torch.log(pred+eps)-torch.log(gt+eps) 

        p1 = torch.mean(d**2)      
        p2 = (d.sum()**2/(d.shape[0]**2))*self.lamda

        return p1-p2


# In[ ]:


class BerhuLoss(nn.Module):
    def __init__(self, scale = 0.5):
        super(BerhuLoss, self).__init__()
        self.scale = scale

    def forward(self, pred, gt):

        pred = pred.view(-1)
        gt = gt.view(-1)

        diff = torch.abs(pred-gt)
        c = 0.2*torch.max(diff)

        l1 = nn.L1Loss()
        mae = l1(pred,gt)

        l2 = nn.MSELoss()
        mse = l2(pred,gt)
        scaledL2 = (mse+(c**2))/(2*c)

        if mae<=c:
              return mae
        else:
              return scaledL2


# In[ ]:


def loss_function(param='l1'):
    if(param=='scale_invariant'):
        loss_fn = ScaleInvariantL2()
    elif(param=='berhu'):
        loss_fn = BerhuLoss()
    elif(param=='l2'):
        loss_fn = nn.MSELoss()
    else:    
        loss_fn = nn.L1Loss()
    return loss_fn


# In[ ]:

param = 'l2'
loss_fn = loss_function(param)

# In[ ]:

def save_models(epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, out+"checkpoint_{}.pt".format(epoch)
    )
    print("Checkpoint saved at epoch ",epoch)


# In[ ]:


# Training Function 
def train(start, num_epochs): 

    print("[INFO] Begin training...")
    startTime = time.time()

    for epoch in range(start, num_epochs): 
        running_train_loss = 0.0 
        running_val_loss = 0.0 
        total = 0.0 

        model.train()
 
        # Training Loop 
        for i, data in enumerate(trainloader): 
            inputs, outputs = data  
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                predicted_outputs = model(inputs)   
                train_loss = loss_fn(predicted_outputs, outputs)    
            running_train_loss += train_loss.item()
#             train_loss.backward() 
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
#             optimizer.step()  
            if((i+1)%step==0):
                scheduler.step()
                
        # Calculate training loss value 
        avg_train_loss = running_train_loss/len(trainloader) 
        
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for i, data in enumerate(valloader): 
                inputs, outputs = data 
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    predicted_outputs = model(inputs) 
                    val_loss = loss_fn(predicted_outputs, outputs) 
                running_val_loss += val_loss.item()  
 
        # Calculate validation loss value 
        avg_val_loss = running_val_loss/len(valloader) 
         
        # Print the statistics of the epoch 
        print('Completed training epoch ', epoch+1)
        wandb.log({ 'Epoch' : epoch+1, 'Training' : avg_train_loss, 'Validation' : avg_val_loss })

        if (epoch+1)%gap_save==0:
            save_models(epoch+1)

    endTime = time.time()
    print("[INFO] End training... \nTime Taken: ",endTime-startTime)


# In[ ]:


model = SUMNet()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# model= nn.DataParallel(model,device_ids = [0, 1]) #if multiple GPUs available
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.98)
step = len(trainloader)//BATCH_SIZE//10


# In[ ]:

wandb.watch(model)
scaler = torch.cuda.amp.GradScaler()
start = 0
train(start,num_epochs)

