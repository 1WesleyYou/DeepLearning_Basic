import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation
import random

# 初始化gym条件，满足项目要求
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

q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
memory = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99  # discount factor
epsilon = 1.0  # 贪心算法系数，用于表示 探索 和 利用 的概率比例

num_episodes = 1000

for episode in range(num_episodes):
    # 进行多轮游戏的迭代
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # ε-贪心算法用于动作选择
        if np.random.rand() < epsilon:
            # 探索, 即采用随机操作
            action = env.action_space.sample()
        else:
            # 利用，已知概率之后执行最好的操作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            # 找到获得最大收益的操作数据
            action = torch.argmax(q_values).item()

        # 执行操作, 更新环境
        next_state, reward, done, _, _ = env.step(action)

        # 这里将获得的数据放进记忆序列，同时剔除过老的数据
        memory.add(state, action, reward, next_state, done)

        # 更新系统状态
        state = next_state
        total_reward += reward

        # 训练网络使用样本输入
        if len(memory.memory) >= batch_size:
            # 从回放缓冲器中获取 样本并且获取经验
            transitions = memory.sample(batch_size)
            batch = np.array(transitions, dtype=object).transpose()
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

            # 转化成张量
            state_batch = torch.FloatTensor(np.stack(state_batch))
            action_batch = torch.LongTensor(np.array(action_batch, dtype=np.int32))
            # action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(np.array(reward_batch, dtype=np.float32))
            # reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(np.stack(next_state_batch))
            not_done_mask = torch.ByteTensor(~np.array(done_batch, dtype=np.uint8))  # 布尔张量， ～ 表示取反

            current_q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))
            next_max_q_values = q_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (gamma * next_max_q_values * not_done_mask)

            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 衰减 ε, 也就是学的越多，应用能力越强
    if epsilon > 0.05:
        epsilon *= 0.995

    print(f'Episode {episode}, Total Reward: {total_reward}')

# 创造图像
fig, ax = plt.subplots()


# 评估训练结果
def test_agent(env, trained_agent):
    frames = []
    state = env.reset()[0]
    done = False
    while not done:
        # Render to RGB array and append to frames
        frames.append(env.render())

        # Choose action
        with torch.no_grad():
            q_values = trained_agent(torch.FloatTensor(np.array(state)).unsqueeze(0))
            action = torch.argmax(q_values).item()

        # Take action
        state, _, done, _, _ = env.step(action)

    env.close()
    return frames


# 动画演示（采用源代码）

def animate_frames(frames):
    img = plt.imshow(frames[0])  # initialize image with the first frame

    def update(frame):
        img.set_array(frame)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    plt.axis('off')
    plt.show()


trained_agent = q_network  # Replace with your trained Q-network
frames = test_agent(env, trained_agent)
animate_frames(frames)
