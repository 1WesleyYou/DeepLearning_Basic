import gym
import torch.nn as nn
import torch
import random
from collections import deque

env = gym.make('CartPole-v1', render_mode='rgb_array')


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):  # 状态和行为分别为输入输出尺寸
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_size, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        x = self.out(x)
        return x


class ReplayBuffer:
    # 存储和采样经验回放数据，以用于训练神经网络
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # 双队列存储数据，不超过容量上限 capacity

    def add(self, state, action, reward, next_state, done):
        # 向回放缓冲器中添加一条经验数据
        # state, action, reward 分别表示状态、动作、奖励数据
        # done 表示是否完成
        self.memory.append((state, action, reward, next_state, done))  # 将这个指令加到记忆序列的最后，如果超过了最大的值就会剔除最老的记忆

    def sample(self, batch_size):
        # 随机抽取一组数据用于神经网络训练
        return random.sample(self.memory, batch_size)


state_size = env.observation_space.shape[0]
action_size = env.action_space.n
