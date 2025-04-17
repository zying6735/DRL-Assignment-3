import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import collections

# --- Dueling DQN Model ---
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

# --- Agent ---
class Agent(object):
    def __init__(self):
        self.device = torch.device("cpu")
        self.action_count = 12
        self.model = DQN(self.action_count).to(self.device)
        self.model.load_state_dict(torch.load("mario_dqn_best.pth", map_location=self.device))
        self.model.eval()

        # For stacking frames manually
        self.frame_stack = collections.deque(maxlen=4)

    def preprocess(self, obs):
        # obs shape: (240, 256, 3) -> resize to (84, 84), grayscale
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        return obs_resized  # shape (84, 84)

    def act(self, observation):
        processed = self.preprocess(observation)

        # Initialize stack if empty (first frame)
        if len(self.frame_stack) == 0:
            for _ in range(4):
                self.frame_stack.append(processed)
        else:
            self.frame_stack.append(processed)

        # Stack frames: shape (4, 84, 84)
        state = np.stack(self.frame_stack, axis=0).astype(np.float32) / 255.0
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)  # shape (1, 4, 84, 84)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action
