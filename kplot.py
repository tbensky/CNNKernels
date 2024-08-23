import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from torchvision import utils
import sys



k = torch.load(sys.argv[1])
print(k.tolist())
#plt.imshow(k.tolist())
plt.imshow(  k.permute(1, 2, 0)  )
plt.show()