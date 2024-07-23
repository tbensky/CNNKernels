# Interpretability of convolutional neural networks via kernel visualizations

The purpose of this work is to do an exercise in the intepretability of a convolutional neural network, by inspecting the evolution of the kernels during training.

The idea for this work came from a study like [this one](https://towardsdatascience.com/convolution-neural-network-decryption-e323fd18c33). There are many such studies.  Like them, our goal is to produce images of kernels of a trained CNN, and see how the may relate to the training data set.

In sum, we wanted to present a convolutional neural network (CNN) with some simple images, get it all trained, then visualize the resulting kernels.

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

We choose these because their features are distinct and simple: corners, vertical/horizontal lines, and gradual slopes.





 