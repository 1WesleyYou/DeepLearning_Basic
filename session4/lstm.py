import numpy as np
import torch.nn as nn
import torch

# 产生正弦波动数据
seq_length = 20
num_samples = 1000

time_steps = np.linspace(0, np.pi, seq_length + 1)  # 在 [0, \pi] 上分为 20 区间
data = np.sin(time_steps)
data = data[:-1].reshape(-1, seq_length)  # -1 表示自动计算维度, 这里删掉了最后一个数据然后变形

# 复杂化数列
x = np.repeat(data, num_samples, axis=0)
y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))

# 转变成张量
x_tensor = torch.FloatTensor(x).unsqueeze(2)  # 能让处理数据的时候更加灵活
y_tensor = torch.FloatTensor(y)


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
        return out
