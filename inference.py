import torch
import torch.nn as nn
from NeuralNet import NeuralNet
import torchvision
import torchvision.transforms as transforms


input_size = 784
hidden_size = 500
hidden_size2 = 350
num_classes = 10
batch_size = 101

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                  transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# model = NeuralNet()
model = NeuralNet(input_size, hidden_size, hidden_size2, num_classes).to(device)
model.load_state_dict(torch.load('model.ckpt', weights_only=True))
model.eval()  # 设置为推理模式


correct = 0
total = 0
image_index_to_display = 8

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        # 如果当前图像是要查看的图像
        # 处理图像和标签
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #print(outputs.shape)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0) # get the batch size
        correct += (predicted == labels).sum().item() # accumulate the corect prediction.

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')


