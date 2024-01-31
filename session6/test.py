import gym
from model import QNetwork, state_size, action_size
import torch
import numpy as np

test_model = QNetwork(state_size, action_size)
test_model.load_state_dict(torch.load('/home/whoami/Desktop/DeepLearning_Basic/pythonProject/session6/model.pth'))

env = gym.make('CartPole-v1', render_mode="human")

done = False
state = env.reset()[0]

while True:
    # Render to RGB array and append to frames
    env.render()

    # Choose action
    with torch.no_grad():
        q_values = test_model(torch.FloatTensor(np.array(state)).unsqueeze(0))
        action = torch.argmax(q_values).item()

    # Take action
    state, _, done, _, _ = env.step(action)

print("game over!")

env.close()
