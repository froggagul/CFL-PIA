import torch                                
import torchvision.transforms as transforms 
import torchvision.datasets as datasets     
import numpy as np                          
import os                                   
import pickle                               
import json                                 
                                            
                                            
def load_cifar10_with_attrs():
    #dataset
    data_loc = "./_data/cifar10"
    n_classes = 100
    class_num = n_classes


    data_transform  = dict()
    data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])



    dataset_train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=data_transform)
    dataset_test = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=data_transform)
    print("CIFAR10")
        


    X=[]
    Y=[]
    for i in range(len(dataset_train)):
        X.append(dataset_train[i][0].numpy())
        Y.append(dataset_train[i][1])


    for i in range(len(dataset_test)):
        X.append(dataset_test[i][0].numpy())
        Y.append(dataset_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    return X, Y, Y
                                           
def load_cifar100_with_attrs():
    #dataset
    data_loc = "./_data/cifar100"
    n_classes = 100
    class_num = n_classes


    data_transform  = dict()
    data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])



    dataset_train = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=data_transform)
    dataset_test = datasets.CIFAR100(root=data_loc, train=False, download=True, transform=data_transform)
    print("CIFAR100")
        


    X=[]
    Y=[]
    for i in range(len(dataset_train)):
        X.append(dataset_train[i][0].numpy())
        Y.append(dataset_train[i][1])


    for i in range(len(dataset_test)):
        X.append(dataset_test[i][0].numpy())
        Y.append(dataset_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    return X, Y, Y