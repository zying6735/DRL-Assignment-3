import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# --- Dueling DQN Model Definition ---
class DQN(nn.Module):
    def __init__(self, action_count):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.adv = nn.Linear(512, action_count)
        self.val = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# --- Required Agent Wrapper for Evaluation ---
class Agent(object):
    def __init__(self):
        self.device = torch.device("cpu")  # ✅ Must use CPU for submission
        self.action_count = 12
        self.model = DQN(self.action_count).to(self.device)
        self.model.load_state_dict(torch.load("mario_dqn_best.pth", map_location=self.device))
        self.model.eval()

    def act(self, observation):
        # observation shape: (4, 84, 84) as a stacked grayscale frame
        state = np.array(observation, dtype=np.float32) / 255.0  # normalize
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action
