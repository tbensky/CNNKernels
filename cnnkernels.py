#!/usr/bin/env -S python3 -u 


import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
import json
import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils
import os
import torch.nn.functional as F
import time
from datetime import datetime
import glob
from torch.nn.functional import normalize

#https://discuss.pytorch.org/t/error-while-running-cnn-for-grayscale-image/598/2
class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.conv_layer_count = 5
        self.K = 10

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_layer_count, kernel_size=self.K, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.01)
        self.fc1 = nn.Linear(67*93, 4096) 
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp1(x)

        #print(x.size()) 
        #exit()

        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
       # x = self.dropout2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)

        return x



    def get_conv_layer_count(self):
        return self.conv_layer_count

    def run_conv1(self,x):
        return self.conv1(x)

    def get_conv1(self):
        return self.conv1.weight


class CustomData(Dataset):
    def __init__(self, dir):
        files = glob.glob(f"{dir}/image*.jpg")
        self.data_len = len(files)
        self.dir = dir
        print(f"{self.data_len} data pairs in {dir}")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        input_file = f"{self.dir}/image_{idx:05d}.jpg"
        img = plt.imread(input_file)
        img_tensor = torch.tensor(img,dtype=torch.float32)
    
        #add the channel_count dimension since conv2d wants [channel_count H W]
        input = img_tensor.unsqueeze(0)
       
        output_file = f"{self.dir}/output_{idx:05d}.dat"
        with open(output_file,"r") as f:
                target = torch.tensor(json.load(f))

        return input,target

    def len(self):
        return self.data_len        

def find_speed():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def plot_kernels(file,img_count):
    #https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
    #kernels = ann.conv1.cpu().weight.detach().clone()   
    #print(ann.state_dict)
    #exit()
    kernels = ann.state_dict()['conv1.weight']
    kernels = kernels.cpu()
    # kernels = ann.conv1.weight.detach().clone()    
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img =utils.make_grid(kernels, nrow = 10)
    # change ordering since matplotlib requires images to 
    # # be (H, W, C)
    plt.imshow(filter_img.permute(1, 2, 0))
    plt.title(f"Epoch {img_count:05d}")
    #plt.savefig(f"Plots/conv_{img_count:05d}.png",dpi=300)
    plt.savefig(file,dpi=300)
    plt.close()

    for (i,k) in enumerate(kernels):
        torch.save(k,f"Kernels/kernel_{img_count:05d}_{i:02d}.pt")


def count_correct(out,target):
    correct = 0
    for i in range(len(out)):
        c = 0
        for j in range(len(out[i])):
            d = torch.abs(out[i][j] - target[i][j])
            if d < 0.1:
                c += 1
        if c == len(out[i]):
            correct += 1
    return correct


device = find_speed()
print(device)
ann = neural_net()
ann.to(device)



########################
## learning rate here ##
########################

#lr=2 or 1.5 reveals interesting features, but loss=nan
#plan: try lr between 1.5 and 15

#Seq01: first working one: lr=0.005, momentum=1.0, dropout=0.25, normalize output, K=20, conv_layer=50, CrossEntropyLoss


#Notes

#22Jul: lr=0.5, momentum=0 gives some kernel patterns
#having variable amplitudes of pulses helps

#24Jul: keep dropout low, like 0.01. lr=0.5, momentum=0, conv_layer=5, no dropout on fc layers!
#conv_layers=10, lower lor=0.001, dropout=0: no good results. conv_layers too large?
#upping conv_layers. Sometimes just be patientand let it run


#25Jul: conv_layers=10, lr=0.05, momentum=0.0 was converging. No great patters in kernels though.

optimizer = optim.SGD(ann.parameters(),lr=0.01) #momentum=0.1)


#CrossEntropyLoss reveals curved sections
#loss_fn = nn.CrossEntropyLoss() 

#MSELoss reveals straight sections
#loss_fn = nn.MSELoss()
#loss_fn = nn.BCEWithLogitsLoss()  #this one is interesting
loss_fn = nn.L1Loss()
#loss_fn = nn.NLLLoss()

#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
train = CustomData("Data")
test_data = CustomData("TestData")
batch_size = int(train.len()/10)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data,shuffle=True)
print(f"data set length={train.len()}, batch_size={batch_size}, test data length={test_data.len()}")


# for i in range(test_data.len()):
#     print(i)
#     test_data.show(i,f"Pulses/pulse_{i:03d}.jpg")
# exit()

os.system("rm Plots/*.png")
os.system("rm loss.csv")
os.system("rm loss.png")
os.system("rm Kernels/*.pt")

epoch = 0
img_count = 0
loss_track = [] # {"epoch": [], "loss": [], "train_correct": [],"test_correct": []}
es = time.time()

#grab random first kernels
plot_kernels(f"Plots/conv_{img_count:05d}.png",img_count)
img_count += 1

while True:
    loss_total = 0.0

    train_correct = 0
    for data,target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
     
        out = ann(data)
   
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

        train_correct += count_correct(out,target)

    test_correct = 0
    for data,target in test_loader:
        data, target = data.to(device), target.to(device)
        out = ann(data)
        test_correct += count_correct(out,target)

    loss_track.append({"epoch": epoch, "loss_total": loss_total, "train_correct": train_correct, "test_correct": test_correct, "dt": datetime.now()}) 
    
    if epoch % 1 == 0:
        ee = time.time()
        print(f"epoch={epoch},loss={loss_total}, train_correct={train_correct}, test_correct={test_correct}, time={ee-es} sec, {datetime.now()}")
        es = ee

        plot_kernels(f"Plots/conv_{img_count:05d}.png",img_count)
        img_count += 1
    
        x = [lt['epoch'] for lt in loss_track]

        plt.subplot(3,1,1)
        y = [lt['loss_total'] for lt in loss_track]
        plt.plot(x,y)
        plt.ylabel("Loss")
        plt.xticks([])
        plt.title("Convolutional Network Training")

        plt.subplot(3,1,2)
        y = [lt['train_correct'] for lt in loss_track]
        plt.plot(x,y)
        plt.xticks([])
        plt.ylabel("Train Correct")

        plt.subplot(3,1,3)
        y = [lt['test_correct'] for lt in loss_track]
        plt.plot(x,y)
        plt.ylabel("Test Correct")
        plt.xlabel("Epoch")
        plt.xticks()

        plt.savefig("loss.png",dpi=300)
        plt.close()

        with open("loss.csv","w") as f:
            f.write("epoch,loss,train_correct,test_correct\n")
            for lt in loss_track:
                f.write(f"{lt['epoch']},{lt['loss_total']},{lt['train_correct']},{lt['test_correct']}\n")

    epoch += 1

    #if correct > 0.95*len(targets):
    if loss_total < 1e-10:
        break

print(f"epoch={epoch},loss={loss_total}")

