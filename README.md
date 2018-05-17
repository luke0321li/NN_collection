# LeNet_tf
The classic <a href="http://yann.lecun.com/exdb/lenet/">LeNet</a> architecture, using tensorflow. MNIST data also included.

## Notes
The layer organization is:
- A 5x5 convolutional layer, from 1 to 6 channels
- A pooling layer reducing the image size by half
- Another 5x5 convolutional layer, from 6 to 16 channels
- Another pooling layer (same as above)
- 3 Fully connected layers, eventually reducing the output dimension to 10

All activation functions are changed to ReLu.
