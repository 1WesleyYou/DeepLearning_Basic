import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim

# Data transformation (converting to Tensor and normalizing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载、加载数据集
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# 下载加载数据集
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)  # shuffle 表示打乱数据顺序


class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 定义正向传播
        # view 将张量 x 变成 1 维度向量
        x = x.view(-1, 28 * 28)
        # 第一个连接层
        x = torch.relu(self.fc1(x))
        # 第二个连接层
        x = torch.relu(self.fc2(x))
        # 第三个连接层
        x = self.fc3(x)
        return x


# 定义神网
model = MNISTNN()
# 定义损失函数为 交叉熵
criterion = nn.CrossEntropyLoss()
# 定义优化器为 SGD
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练一整个模型
for epoch in range(5):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # 去除权重矩阵原有的梯度
        optimizer.zero_grad()

        # 正向传播到底 + 反向传播 + 优化
        outputs = model(inputs)
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 损失可视化
        if i % 500 == 0:
            print(f'Epoch [{epoch + 1}/5], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

correct = 0
total = 0
with torch.no_grad():
    # 评估我们的模型
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
