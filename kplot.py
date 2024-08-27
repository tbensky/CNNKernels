import torch
import sys

k = torch.load(sys.argv[1])
print(k.tolist())
#plt.imshow(k.tolist())
plt.imshow(  k.permute(1, 2, 0)  )
plt.show()