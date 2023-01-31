import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchdp.per_sample_gradient_clip import PerSampleGradientClipper

from models import ModelFactory, Model

def conv_shape(x, k, p=0, s=1,
               d=1):  # x=dim_input, p=padding, d=dilation, k=kernel_size, s=stride # convolution 차원 계산 함수
    return int((x + 2 * p - d * (k - 1) - 1) / s + 1)

def calculate_shape(init):  # 최종 output 차원 계산
    size_1 = conv_shape(init, 3)
    size_2 = conv_shape(size_1, 2, 0, 2)
    size_3 = conv_shape(size_2, 3)
    size_4 = conv_shape(size_3, 2, 0, 2)
    size_5 = conv_shape(size_4, 3)
    size_6 = conv_shape(size_5, 2, 0, 2)
    return size_6

class cnn_feat_extractor(nn.Module):  # CNN 모델
    def __init__(self, input_shape=(3, 50, 50), n=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, n, 3)
        self.pool3 = nn.MaxPool2d(2)

        size_a = calculate_shape(input_shape[1])
        size_b = calculate_shape(input_shape[2])
        self.fc1 = nn.Linear(n * size_a * size_b, 256)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x

def nm_loss(pred, label):  # loss 정의
    loss = F.cross_entropy(pred, label)

    return torch.mean(loss)

@ModelFactory.register('nm_cnn')
class NMCNN(Model):  # 최종 classifier, optimizer 등
    def __init__(self, classes=2, input_shape=(3, 50, 50), lr=0.01, n=128,args=None):
        super().__init__()
        self.fe = cnn_feat_extractor(input_shape, n)
        self.fc2 = nn.Linear(256, classes)
        self.criterion = nm_loss
        self.optimizer = optim.SGD(self.parameters(), lr)
        if args.ldp:
            self.clipper = PerSampleGradientClipper(self,args.clip)
            #self.criterion = nn.CrossEntropyLoss()#(reduction='none')
            
    def forward(self, x):
        x = self.fe(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x

@ModelFactory.register('nm_cnn')
class NMCNN(Model):  # 최종 classifier, optimizer 등
    def __init__(self, classes=2, input_shape=(3, 50, 50), lr=0.01, n=128,args=None):
        super().__init__()
        self.fe = cnn_feat_extractor(input_shape, n)
        self.fc2 = nn.Linear(256, classes)
        self.criterion = nm_loss
        self.optimizer = optim.SGD(self.parameters(), lr)
        if args.ldp:
            self.clipper = PerSampleGradientClipper(self,args.clip)
            #self.criterion = nn.CrossEntropyLoss()#(reduction='none')
            
    def forward(self, x):
        x = self.fe(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x
