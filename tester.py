import torch
import torch.optim as optim
import torch.cuda as cuda
import model
import dataloader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

net=model.ResNet56()
device=cuda.device('cuda:0,1' if torch.cuda.is_available() else print('error'))
net.to('cuda:0,1')

file_name = 'resnet56_cifar10.pt'
criterion=nn.CrossEntropyLoss()
rate=0.1
optimizer = optim.SGD(net.parameters(),lr=rate, momentum=0.9, weight_decay=0.0001)
test_accuracy=0.0
max5=np.zeros(5)

def train(epoch):

    #print("Current train Epoch: ", epoch+1)
    running_loss=0.0
    correct=0.0
    total=0.0

    for i,data in enumerate(dataloader.trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss+= loss.item()              
        _, predicted = outputs.max(1)                       #Get the predicted output using the model
           
        correct += predicted.eq(labels).sum().item()        #Calculate how many ouput values are correct
        total += labels.size(0)                             # total number of input/output datasets
        if i%100==0:
            pass
            #print("\nCurrent batch: ",i+1)
            #print("Current train loss: ", loss.item())
            #print("Current train accuracy: ", (predicted.eq(labels).sum().item()/labels.size(0)),"\n")


    #print("Total train loss:", running_loss)
    
    #print("Total train accuracy:", (correct/total),"\n\n")



def test(epoch):

    print("Current test epoch: ",epoch+1)
    running_loss=0.0
    correct = 0.0
    total = 0.0
 

    for i,data in enumerate(dataloader.testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            outputs = net(inputs)

            loss=criterion(outputs, labels)
            running_loss+= loss.item()              
            _, predicted = outputs.max(1)                      # Get the predicted output using the model
            

            correct += predicted.eq(labels).sum().item()       # Calculate how many ouput values are correct
            total += labels.size(0)                            # total number of input/output datasets
            test_accuracy=(correct/total)

    print("Total test loss:", running_loss)
    print("Total test accuracy:", test_accuracy,"\n\n")


def learning_rate(optimizer,epoch):
    if epoch<82:
        lr=0.1
    elif epoch<109:
        lr=0.01
    else:
        lr=0.001

    for param_group in optimizer.param_groups:
         param_group['lr'] = lr


    
for epoch in range(164):
    learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
    if test_accuracy>max5[0]:
        max5[0]=test_accuracy
        max5.sort()
    

print("Top1 accuracy:", max5[4])
print("Top5 accuracy:", np.sum(max5)/5)    
print("Completed")    
