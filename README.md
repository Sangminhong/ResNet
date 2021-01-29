# ResNet
Deep Residual network implementation using CIFAR-10 dataset.


### The Original work
[Deep Residual Learning for Image Recognition (the 2015 ImageNet competition winner)](https://arxiv.org/abs/1512.03385)


### Residual blocks
Resudiual block consists of Basic block structure and Bottleneck block structure. In this paper, the basic block structure is prooposed for CIFAR-10 dataset. 

![alt_text](https://github.com/Sangminhong/ResNet/blob/main/Images/basicblock%20and%20bottleneck%20block.PNG)<br/>
Figure: Left figure shows Basic block structure and right figure illustrates the Bottleneck block structure

### CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images.
[Download] https://www.cs.toronto.edu/~kriz/cifar.html



### Implementation
![alt text](https://github.com/Sangminhong/ResNet/blob/main/Images/Architecture%20summary.PNG)<br/>
Figure: Architecture summary for Deep Residual network using CIFAR-10 dataset
Weight decay: 0.0001<br/>
Momentum: 0.9<br/>
Batch size: 128<br/>
Learning rate: 0.1->0.01->0.001<br/>
Data augmentatation: 4pixels padded on each side, 32*32 random crop, horizontal flip.<br/>
