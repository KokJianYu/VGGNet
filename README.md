# VGGNet

This project is an implementation of the VGGNet that was introduced in the paper "Very Deep Convolutional Networks For Large-Scale Image Recognition" [Link to paper](https://arxiv.org/pdf/1409.1556.pdf)

# Prerequisite

Required libraries can be found in requirements.txt.

Install with the following command:
`pip install -r requirements.txt`

# Dataset
CIFAR10

# Model
I have slightly modified the model to speed up the training process.

For the fully connected layers, instead of having 4096 units, I have changed it to 512 units.

I have also included batch normalization layer to help speed up the training process.

# Training and Evaluation
The code to train and test the implementation is included in `main.ipynb`.




