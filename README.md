# NN_collection
A collection of Neural Network architectures. MNIST data also included.

## LeNet
The classic <a href="http://yann.lecun.com/exdb/lenet/">LeNet</a> architecture, using tensorflow. 
The layer organization is:
- A 5x5 convolutional layer, from 1 to 6 channels
- A pooling layer reducing the image size by half
- Another 5x5 convolutional layer, from 6 to 16 channels
- Another pooling layer (same as above)
- 3 Fully connected layers, eventually reducing the output dimension to 10

All activation functions are changed to ReLu.


## Autoencoder (AE)
An autoencoder for creating low-dimensional latent representations of data.
The layer organization is:
### Encoder
- A fully connected layer from 784 to 100 nodes
- Another fully connected layer from 100 nodes to the custom dimension of latent representation
### Decoder
- A fully connected layer from the latent representation to 100 nodes
- A final fully connected layer from 100 to 784 nodes


## Variational Autoencoder (VAE)
Like the AE but the latent representation now comes in the form of a mean vector and a standard deviation vector (of a multidimensional gaussian with diagonal covariance matrix)
### Encoder
- A 784->784 fully connected layer
- Another fully connected layer creating k-dimensional mean and standard deviation vectors
### Sampler
- Draws a batch of samples (k-dimensional vectors) from a multidimensional gaussian with the above generated mean and standard deviation
### Decoder
- A k->784 fully connected layer
- Another 784->784 fully connected layer to recreate the samples


## Neural Arithmetic Logic Unit (NALU)
Capable of learning simple arithmetic expressions and "generalize to numbers that are several orders of magnitude larger than those observed during training", described in <a href="https://arxiv.org/abs/1808.00508">this paper</a>
