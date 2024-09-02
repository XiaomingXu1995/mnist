import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Fully connected neural network with one hidden layer
#定义基础模型都继承自nn.Module
class NeuralNet(nn.Module):
#定义神经网络的结构，包含输入层大小、隐藏层大小和输出层类别数。
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
#定义线性层，全连接层，包含输入维度，输出维度
        self.layer1 = nn.Linear(input_size, hidden_size) 
#可以搜一下ReLU激活函数以及各激活函数的功能以及作用，max(0,x)说实话我一直不太理解这个吊函数
        self.layer2 = nn.ReLU()

        self.layer3 = nn.Linear(hidden_size, hidden_size2)
        self.layer4 = nn.ReLU()
        
        self.layer5 = nn.Linear(hidden_size2, num_classes)
#前向传播方法，定义了神经网络的前向传播过程。 这里就是简单的第一个全连接层接收输入，然后通过激活函数，结果进入第二个全连接层
#可以简单理解为 y = w2*（Relu（w1*x+b1））+b2
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
