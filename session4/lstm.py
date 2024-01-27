import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

# 产生正弦波动数据
seq_length = 20
num_samples = 1000


# 由于这里的sin波生成函数会重复使用，所以改为使用函数定义，输入一个数据数量，结果是随机的
def sin_gen(num_step):
    time_steps = np.linspace(0, np.pi, seq_length + 1)  # 在 [0, \pi] 上分为 20 区间
    data = np.sin(time_steps)
    data = data[:-1].reshape(-1, seq_length)  # -1 表示自动计算维度, 这里删掉了最后一个数据然后变形

    # 复杂化数列
    x = np.repeat(data, num_samples, axis=0)
    y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))

    # 转变成张量
    x_tensor_otp = torch.FloatTensor(x).unsqueeze(2)  # 能让处理数据的时候更加灵活
    y_tensor_otp = torch.FloatTensor(y)
    return x_tensor_otp, y_tensor_otp


x_tensor, y_tensor = sin_gen(num_samples)


# 定义 lstm 网络模型
class LstmNNModel(nn.Module):
    def __init__(self):
        super(LstmNNModel, self).__init__()
        self.lstm = nn.LSTM(1, 50)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        hide_ini = torch.zeros(1, x.size(1), 50)  # 初始化 隐藏层， 内容为 0, x.size(1) = batch_size
        cell_ini = torch.zeros(1, x.size(1), 50)

        out, _ = self.lstm(x, (hide_ini, cell_ini))  # 丢弃返回的隐藏状态单元，保留 x (seq_len, batch_size, hidden_size) 的张量
        out = self.fc(out[:, -1:])  # 这里用out而非x可以让操作更加灵活，解释器不容易报错
        out = out.view(-1, 1)
        return out


# 定义这个模型、优化器
model = LstmNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x_tensor)
    # print(output.size(), y_tensor.size())
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# 测试数据集
num_test_samples = 100
test_x_tensor, test_y_tensor = sin_gen(num_test_samples)

with torch.no_grad():
    predicted = model(test_x_tensor)
    mse = criterion(predicted, test_y_tensor).item()
    print(f'Mean Squared Error on test data: {mse:.4f}')
