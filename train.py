#我看了一下，这是一个torch的手写数字分类器，通过训练一个全连接神经网络模型来实现对手写数字的自动识别，MNIST是经典的手写数字数据集。
#就是给你一张图片自动识别出这是0～9哪一个

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from NeuralNet import NeuralNet


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")
# Hyper-parameters 
input_size = 784
hidden_size = 500
hidden_size2 = 350
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
#加载数据集捏
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

#所有的基本都能看到DataLoader，他是创建训练数据加载器
#batch_size 参数指定每个批次的样本数量，然后一个Epoch要进行ff 和backward的次数就是总样本数/batch_size
#shuffle 参数：是否在每个 epoch 重新打乱数据。我感觉是为了防止过拟合
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


#建神经网络模型，移到指定的设备上。device在前面定义了，就看看能不能检测到gpu
model = NeuralNet(input_size, hidden_size, hidden_size2, num_classes).to(device)

# Loss and optimizer
#交叉熵损失函数，二分类问题经典函数，因为他这个函数求导后很好计算，具体推导我忘了
criterion = nn.CrossEntropyLoss()
#定义了优化器，Adam 优化器，用于优化模型参数。lr就是学习率，可理解为调参的步长
#adam的主要思路我记得是动态调整学习率，因为梯度下降到后期学习率就不能太大了，他会根据参数改动的情况自适应学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
#计算批次的数量，即模型在每个训练周期（epoch）中需要遍历的批次数量。
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        #处理图像，将图像扁平化，弄成模型可接受的输入维度
        images = images.reshape(-1, 28*28).to(device)
        #每个图像都有一个对应的输出label，因为是分类问题
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        #清空梯度
        optimizer.zero_grad()
        #反向传播，重新计算梯度
        loss.backward()
        #参数更新
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
