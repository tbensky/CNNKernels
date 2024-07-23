# Interpretability of convolutional neural networks via kernel visualizations

The purpose of this work is to do an exercise in the intepretability of a convolutional neural network, by inspecting the evolution of the kernels during training.

The idea for this work came from a studies like 

  * [Lee et. al.](https://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf), in particular Figs. 2 and 3, and
  * [Krizhevsky, et. al.](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), Fig. 3.  

Like them, our goal is to visualize the kernels of a trained CNN, and see how they may relate to the training data set.

In sum: we want to train a convolutional neural network (CNN) with some simple images, then visualize the resulting kernels.

# Our data set

To hopefully keep the kernel visualizations simple, we produced a series of 1x100x100 images of basic waveforms one might encounter in an electroics lab: square, triangle, and gaussian waveforms.  Here are a few:

![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_000.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_001.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_002.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_003.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_004.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_005.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_006.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_007.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_008.jpg)

We chose these because:

 1.  Their features are distinct and simple, mostly having corners, vertical/horizontal lines, and gradual slopes. (The waveforms in the entire data all have have varying amplitudes, widths, and vertical offsets.) 
 1.  They have a bit of data/laboratory flair, since our larger goal is to study if CNNs and kernels can be used by those in the basic sciences to gain insights into their data.

 # PyTorch and the neural network

 Since this job is very similar to building a CNN to recognize the MNIST digits, we used that as a base for the structure of our model.  Here is it:

 ```python
 class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.conv_layer_count = 5
        self.K = 20

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_layer_count, kernel_size=self.K, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(31*31, 4096)
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
       
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)


        #x = F.normalize(x)
        return x
```

We're bascially coming in with a CNN having 5 kernels, each with a 20x20 size.  We activate it with a ReLU, then go on into another CNN layer+ReLU, the do some max pooling.  We then flatten it and head into 3 linear, dense, fully connected layers.  Nothing too special here.

Note:
 * We played around a lot with the size and number of kernels.  The 20x20 size seemed like a good



 self.relu = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(31*31, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 3)



 