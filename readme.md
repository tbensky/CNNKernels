# Interpretability of convolutional neural networks via kernel visualizations

The purpose of this work is to do an exercise in the intepretability of a convolutional neural network, by inspecting the evolution of the kernels during training.

The idea for this work came from a studies like 

  * [Lee et. al.](https://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf), in particular Figs. 2 and 3, and
  * [Krizhevsky, et. al.](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), Fig. 3.  

Like them, our goal is to visualize the kernels of a trained CNN, and see how they may relate to the training data set.

In sum: we want to train a convolutional neural network (CNN) with some simple images, then visualize the resulting kernels.

# Our data set

To keep the data set simple, we produced a series of images of simple waveforms one might encounter in an electroics lab: square, triangle, and gaussian waveforms.  Here are a few:

![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_000.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_001.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_002.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_003.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_004.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_005.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_006.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_007.jpg)
![alt text](https://github.com/tbensky/CNNKernels/blob/main/Assets/Pulses/pulse_008.jpg)

We chose these because their features are distinct and simple, mostly having corners, vertical/horizontal lines, and gradual slopes. (The waveforms in the entire data set were created to have varying amplitudes, widths, and vertical offsets.)

x





 