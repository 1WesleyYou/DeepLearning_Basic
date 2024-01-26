import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Training Data
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Testing Data
X_test = torch.tensor([[0.1, 0.0], [0.0, 0.9], [1.1, 0.0], [0.9, 1.0]], dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(SimpleNN, self).__init__()
        # 第一个线性层，输入维度2,输出维度2
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        # torch.sigmoid 函数能对 tensor 中的每一个变量代入 sigmoid 中处理
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


# 实例化
model = SimpleNN()
# 采用均方差标准
criterion = nn.MSELoss()
# 定义优化器，及最小化损失的算法，这里使用学习率 0.1 的随机梯度下降 SGD
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练 10000 次，epoch 中文叫 周期
for epoch in range(10000):
    # 正向传播，输入训练数据，得到整个模型的输出
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    # 多次反向传播前需要将累积的梯度清零
    optimizer.zero_grad()
    # 计算损失对于模型参数的梯度，找到梯度方向
    loss.backward()
    # 用于执行优化步骤
    optimizer.step()

    # Print Loss
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
