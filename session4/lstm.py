import numpy as np
import torch

# 产生正弦波动数据
seq_length = 20
num_samples = 1000

time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data = data[:-1].reshape(-1, seq_length)

# 复杂化数列
x = np.repeat(data, num_samples, axis=0)
y = np.sin(time_steps[-1] * np.ones((num_samples, 1)))

# 转变成张量
x_tensor = torch.FloatTensor(x).unsqueeze(2)
y_tensor = torch.FloatTensor(y)
