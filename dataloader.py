from PIL import Image
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot
import torchvision.transforms as tr
import torch.nn as nn


class MyDataset (Dataset):


    def __init__(self, x_data, y_data, transform=None):

        
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)
        
        

            
    def __getitem__(self, index):
        inputs, labels = self.x_data[index], np.array(self.y_data[index])
        inputs = Image.fromarray(inputs)

        if self.transform:
            inputs = self.transform(inputs)

        return inputs, labels
 
    def __len__ (self):
        return self.len



class ToTensor:
    def __call__(self, sample):
       inputs, labels = sample
       inputs = torch.from_numpy(inputs)
       inputs = inputs.permute(2,0,1)
       labels = torch.from_numpy(labels)

       return inputs, labels



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




def preprocess_d(batch_data):
    newdata= np.array(batch_data)
    #data=data.reshape(10000,3,1024)
    newdata=newdata.reshape(10000,3,32,32)
    newdata=newdata.transpose((0,2,3,1))
    return newdata
 


def preprocess_l(batch_labels):
    newlabels=np.array(batch_labels)
    return newlabels





batch1=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_1")
batch2=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_2")
batch3=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_3")
batch4=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_4")
batch5=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_5")
test_batch=unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/test_batch") 


data1=preprocess_d(batch1[b'data'])
data2=preprocess_d(batch2[b'data'])
data3=preprocess_d(batch3[b'data'])
data4=preprocess_d(batch4[b'data'])
data5=preprocess_d(batch5[b'data'])


labels1 = preprocess_l(batch1[b'labels'])
labels2 = preprocess_l(batch2[b'labels'])
labels3 = preprocess_l(batch3[b'labels'])
labels4 = preprocess_l(batch4[b'labels'])
labels5 = preprocess_l(batch5[b'labels'])



train_images=np.concatenate((data1,data2,data3,data4,data5))
#train_images = train_images.astype(np.float32)
train_labels=np.concatenate((labels1,labels2,labels3,labels4,labels5))
train_labels = train_labels.astype(np.long)
test_images=preprocess_d(test_batch[b'data'])
#test_images = test_images.astype(np.float32)
test_labels = preprocess_l(test_batch[b'labels'])
test_labels = test_labels.astype(np.long)



trans = tr.Compose([
     tr.RandomCrop(32, padding=4),   # crop the given image at a random location
     tr.RandomHorizontalFlip(),      # Horizontally flip the given image randomly with given probability. default=0.5
     
     tr.ToTensor(),
     tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 ])  

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)

trainset = MyDataset(train_images,train_labels, transform=trans)
testset = MyDataset(test_images, test_labels, transform=trans)



trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

#first_data = trainset[0]
#features, labels = first_data
