
from PIL import Image
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot
import torchvision.transforms as tr
import torch.nn as nn
import pickle

class MyDataset (Dataset):
        

    def __init__(self, train, test, transform=None):

        batch1=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_1")
        batch2=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_2")
        batch3=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_3")
        batch4=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_4")
        batch5=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/data_batch_5")
        test_batch=self.unpickle("/home/mchiash2/ResNET/data/cifar-10-batches-py/test_batch") 


        data1=self.preprocess_d(batch1[b'data'])
        data2=self.preprocess_d(batch2[b'data'])
        data3=self.preprocess_d(batch3[b'data'])
        data4=self.preprocess_d(batch4[b'data'])
        data5=self.preprocess_d(batch5[b'data'])


        labels1 = self.preprocess_l(batch1[b'labels'])
        labels2 = self.preprocess_l(batch2[b'labels'])
        labels3 = self.preprocess_l(batch3[b'labels'])
        labels4 = self.preprocess_l(batch4[b'labels'])
        labels5 = self.preprocess_l(batch5[b'labels'])

        self.train_images=np.concatenate((data1,data2,data3,data4,data5))
        #train_images = train_images.astype(np.float32)
        self.train_labels=np.concatenate((labels1,labels2,labels3,labels4,labels5))
        self.train_labels = self.train_labels.astype(np.long)
        self.test_images=self.preprocess_d(test_batch[b'data'])
        #test_images = test_images.astype(np.float32)
        self.test_labels = self.preprocess_l(test_batch[b'labels'])
        self.test_labels = self.test_labels.astype(np.long)

        if test==1:
            self.x_data = self.test_images
            self.y_data = self.test_labels
        elif train==1:
            self.x_data = self.train_images
            self.y_data = self.train_labels
        
        self.transform = transform
        self.len = len(self.y_data)
        
        

            
    def __getitem__(self, index):
        inputs, labels = self.x_data[index], np.array(self.y_data[index])
        inputs = Image.fromarray(inputs)

        if self.transform:
            inputs = self.transform(inputs)

        return inputs, labels
 
    def __len__ (self):
        return self.len

    # def get_train(self):
    #     return self.train_labels

    # def get_testlabel(self):
    #     return self.test_laebls

    def get_train_images(self):
        return self.train_images

    def get_train_labels(self):
        return self.train_labels

    def get_test_images(self):
        return self.test_images

    def get_test_labels(self):
        return self.test_labels


    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def preprocess_d(self,batch_data):
        newdata= np.array(batch_data)
        #data=data.reshape(10000,3,1024)
        newdata=newdata.reshape(10000,3,32,32)
        newdata=newdata.transpose((0,2,3,1))
        return newdata


    def preprocess_l(self,batch_labels):
        newlabels=np.array(batch_labels)
        return newlabels



class ToTensor:
    def __call__(self, sample):
       inputs, labels = sample
       inputs = torch.from_numpy(inputs)
       inputs = inputs.permute(2,0,1)
       labels = torch.from_numpy(labels)

       return inputs, labels








 















