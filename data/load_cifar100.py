import torch                                
import torchvision.transforms as transforms 
import torchvision.datasets as datasets     
import numpy as np                          
import os                                   
import pickle                               
import json                                 
                                            
                                            
                                            
def load_cifar100_with_attrs(args):
    #dataset
    data_loc = args.dataset_dir
    n_classes = args.n_classes
    class_num = n_classes
    p = args.non_iid_p


    data_transform  = dict()
    data_transform["CIFAR100"] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])



    dataset_train = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=data_transform[args.dataset])
    dataset_test = datasets.CIFAR100(root=data_loc, train=False, download=True, transform=data_transform[args.dataset])
    print("CIFAR100")
        


    X=[]
    Y=[]
    for i in range(len(dataset_train)):
        X.append(dataset_train[i][0].numpy())
        Y.append(dataset_train[i][1])
        
    X_test=[]
    Y_test=[]

    for i in range(len(dataset_test)):
        X_test.append(dataset_test[i][0].numpy())
        Y_test.append(dataset_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    X_test=np.array(X_test)
    Y_test=np.array(Y_test)

