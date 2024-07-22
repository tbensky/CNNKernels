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

#https://discuss.pytorch.org/t/error-while-running-cnn-for-grayscale-image/598/2
class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.conv_layer_count = 50
        self.K = 20

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_layer_count, kernel_size=self.K, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(31*31, 128)
        self.fc2 = nn.Linear(128, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.dropout1(x)
    
        #K=10, size=18x18
        #K=25, size=7x7
        #K=20, 10x10
        #print(x.size()) 
        #exit()
       
        x = torch.flatten(x, 1)
        x = self.fc1(x)
   
        x = self.relu(x)
        x = self.fc2(x)

        x = F.normalize(x)
        return x

    def get_conv_layer_count(self):
        return self.conv_layer_count

    def run_conv1(self,x):
        return self.conv1(x)

    def get_conv1(self):
        return self.conv1.weight


class CustomData(Dataset):
    def __init__(self, json_file):
        with open(json_file,"r") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input = self.pairs[idx][0]
        target = self.pairs[idx][1]

        ip = torch.tensor(input).view(100,100)

        #add the channel_count dimension since conv2d wants [channel_count H W]
        ip = ip.unsqueeze(0)

        t = torch.tensor(target)
        #t = F.normalize(t,dim=0)
       
        return ip, t

    def len(self):
        return len(self.pairs)

    def show(self,idx,file):
        p = np.array(self.pairs[idx][0])
        p1 = p.reshape(100,100)
        plt.imshow(p1)
        plt.savefig(file,dpi=300)
        plt.close()
        

def find_speed():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

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

optimizer = optim.SGD(ann.parameters(),lr=1.5,momentum=1.0)


#CrossEntropyLoss reveals curved sections
#loss_fn = nn.CrossEntropyLoss() 

#MSELoss reveals straight sections
loss_fn = nn.MSELoss()
#loss_fn = nn.BCEWithLogitsLoss()

#https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets

size = 100
train = CustomData("pairs.json")
train_loader = DataLoader(train, batch_size=100, shuffle=True)

# for i in range(train.len()):
#     print(i)
#     train.show(i,f"Pulses/pulse_{i:03d}.jpg")
# exit()

os.system("rm Plots/*.png")
os.system("rm loss.csv")
os.system("rm loss.png")

epoch = 0
img_count = 0
loss_track = {"epoch": [], "loss": []}
es = time.time()

while True:
    loss_total = 0.0

    correct = 0
    for data,target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
     
        out = ann(data)
   
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

        correct_list = (torch.abs(out - target) < 0.1)
        if correct_list == [True] * len(target):
            correct += 1
    
    loss_track["epoch"].append(epoch)
    loss_track["loss"].append(loss_total)
    
    if epoch % 5 == 0:
        ee = time.time()
        print(f"epoch={epoch},loss={loss_total}, correct={correct}, time={ee-es} sec")
        es = ee
    
        plt.tight_layout()
    
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
        plt.savefig(f"Plots/conv_{img_count:05d}.png",dpi=300)
        img_count += 1
        plt.close()

        plt.plot(loss_track['epoch'],loss_track['loss'])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Convolutional Network Training Loss")
        plt.savefig("loss.png",dpi=300)
        plt.close()

        with open("loss.csv","w") as f:
            f.write("epoch,loss\n")
            for (epoch,loss) in zip(loss_track['epoch'],loss_track['loss']):
                f.write(f"{epoch},{loss}\n")

    
        

    epoch += 1

    #if correct > 0.95*len(targets):
    if loss_total < 1e-10:
        break

print(f"epoch={epoch},loss={loss_total}")

