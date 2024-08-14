# Interpretability of convolutional neural networks via kernel visualizations

The purpose of this work is to do an exercise in the intepretability of a convolutional neural network, by inspecting the evolution of the kernels during training.

The idea for this work came from a studies like 

  * [Lee et. al.](https://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf), in particular Figs. 2 and 3, and
  * [Krizhevsky, et. al.](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), Fig. 3.  

Like them, our goal is to visualize the kernels of a trained CNN, and see how they may relate to the training data set.

In sum: we want to train a convolutional neural network (CNN) with some simple images, then visualize the resulting kernels.

# Our data set

To hopefully keep the kernel visualizations simple, we produced a series of 1x100x100 images of basic waveforms one might encounter in an electroics lab: square, sinusoid, and gaussian waveforms.  Here are a few:


![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/pulse_montage01.png)

See the [Pulses](https://github.com/tbensky/CNNKernels/tree/main/Assets/SamplePulses/Pulses) folder for 100 such images.

We chose these because:

 1.  Their features are distinct and simple, mostly having corners, vertical/horizontal lines, and gradual slopes. (The waveforms in the entire data all have have varying amplitudes, widths, and vertical/horizontal offsets.) 
 1.  They have a bit of data/laboratory flair, since our larger goal is to study if CNNs and kernels can be used by those in the basic sciences to gain insights into their data. These images could easily be pulled right off of an oscilloscope

 # PyTorch and the neural network

 Since this job is very similar to building a CNN to recognize the MNIST digits, we used that as a base for the structure of our model. We studied [this code](https://github.com/pytorch/examples/blob/main/mnist/main.py) a lot.
 
Here's what we used:

 ```python
 class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.conv_layer_count = 5
        self.K = 25

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.conv_layer_count, kernel_size=self.K, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.mp1 =  nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=self.conv_layer_count, out_channels=1, kernel_size=self.K)
        self.act = nn.Tanh()
        self.dropout1 = nn.Dropout(0.01)
        self.dropout2 = nn.Dropout(0.01)
        self.fc1 = nn.Linear(226*226, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)

        return x
```

We're bascially coming in with a CNN having 5 kernels, each with a 25x25 size.  We activate it with a ReLU, then go on into another CNN layer+ReLU, then some max pooling, and some dropout.  We then flatten it and head into 3 linear, dense, fully connected layers with Tanh activation.  Nothing too special here, and we spent a lot of time 'playing' with the structure of the network until it worked consistently.



# Notes

 * Our images are 100x100 with 1-bit of depth (simple pixel on or off images). They are each to be mapped to a 3-bit binary value, as: 001=square pulse, 010=Gaussian pulse and 100=triangular pulses.

 * We played around a lot with the size and number of kernels.  The 25x25 size seemed like a good size for our 100x100 images with a stride of 1. ("seem like a good size"=we thought a 25x25 square "sliding over" our images would capture the features of our clunky waveform images.)

 * The more kernels we had, the less defined each would be in the end. We tried 100, 50, 20, etc. and think for our simple waveforms, not many kernels are needed. In fact, it may be better to restrict the system more, so each kernel contributes more to the training, bringing out the features we're hoping to see.

 * The size of the fully connected (fc) layers was a big variable too. After working with PiNNS (see our report [here](https://github.com/tbensky/PiNN_Projectile)), we realized that these networks need 'expressivity', which wide layers seem to supply. So, we went for a 4096 into a 1024 into a 3 for the eventual output.

 * L1Loss was the only loss function that seemed to work consistently, but we noticed this is not the most popular one to use with CNNs.

 * I found the network very finicky to train, but this seems to work: L1Loss, Tanh activation on the fully connected layers, lr=0.1, momentum=0.0, batch_size=1,000 (on the 10,000 input images), and very selective DropOut.

* My network is huge (200M parameters). We kind of like using this function to count our parameters

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

as shown [here](https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model).


# Loss Profile

Typical loss profiles that evolves during training are quite fun to see. Here's one:

![loss profile](https://github.com/tbensky/CNNKernels/blob/main/Assets/LossProfiles/loss01.png)

The top graph is the straight up L1-loss. The middle one is the number of training samples the network is able to
recognize at a given epoch. The lower graph is the number of test samples (not in the training set) that
the network can recognize (50 used in the runs shown).

Here's another loss profile:

![loss profile](https://github.com/tbensky/CNNKernels/blob/main/Assets/LossProfiles/loss02.png)

# Results

Here are some results showing what the kernels ended up converging too. The strips of 4, 5 or 10 were from different runs.

![kernels](https://github.com/tbensky/CNNKernels/blob/main/Assets/Kernels/kernels.png)

 


# References

 * [HOW CONVOLUTIONAL NEURAL NETWORKS SEE THE WORLD — A SURVEY OF CONVOLUTIONAL NEURAL NETWORK VISUALIZATION METHODS, by Qin et. al.](https://arxiv.org/pdf/1804.11191)

 * [Kernels on a trained CNN seem random](https://ai.stackexchange.com/questions/46180/kernels-on-a-trained-cnn-seem-random)

 