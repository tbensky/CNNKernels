# Interpretability of convolutional neural networks via kernel visualizations

The purpose of this work is to do an exercise in the intepretability of a convolutional neural network, by inspecting the evolution of the kernels during training.

The idea for this work came from a study like [this one](https://towardsdatascience.com/convolution-neural-network-decryption-e323fd18c33). There are many such studies.  Like them, our goal is to produce images of kernels of a trained CNN, and see how the may relate to the training data set.

In sum, we wanted to present a convolutional neural network (CNN) with some simple images, get it all trained, then visualize the resulting kernels.

# Our data set

To keep the data set simple, we produced a series of images of simple waveforms one might encounter in an electroics lab: square, triangle, and gaussian waveforms.  Here are a few:




 