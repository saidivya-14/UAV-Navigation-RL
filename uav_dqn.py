# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import deque

# # ✅ Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load UAV dataset
# csv_file_path = "D:\\building projects\\uav_project\\UAV_full_data.csv"  
# data = pd.read_csv(csv_file_path)

# # ✅ Define obstacles & assign severity based on dataset
# obstacle_positions = {}
# grid_size = 10  

# for _, row in data.iterrows():
#     try:
#         x, y = int(row['Start Frame']), int(row['End Frame'])
#         severity = row.get("Obstacle Severity", 10)  # Default penalty if missing
#         obstacle_positions[(x % grid_size, y % grid_size)] = severity
#     except ValueError:
#         continue

# if not obstacle_positions:
#     print("⚠️ Warning: No obstacles detected in the dataset!")

# # ✅ Define UAV Environment with Dataset-Based Rewards
# class UAVEnvironment:
#     def __init__(self, grid_size, target, obstacles):
#         self.grid_size = grid_size
#         self.state = (0, 0)  
#         self.target = target  
#         self.obstacles = obstacles  
#         self.actions = ["up", "down", "left", "right"]

#     def reset(self):
#         self.state = (0, 0)
#         return np.array(self.state, dtype=np.float32)

#     def step(self, action):
#         x, y = self.state
#         if action == "up":
#             x = max(x - 1, 0)
#         elif action == "down":
#             x = min(x + 1, self.grid_size - 1)
#         elif action == "left":
#             y = max(y - 1, 0)
#         elif action == "right":
#             y = min(y + 1, self.grid_size - 1)

#         new_state = (x, y)
#         distance_to_target = np.linalg.norm(np.array(new_state) - np.array(self.target))

#         # ✅ **Dataset-Based Reward Calculation**
#         if new_state in self.obstacles:
#             reward = -self.obstacles[new_state]  # Penalty based on obstacle severity
#         elif new_state == self.target:
#             reward = 100  
#         else:
#             reward = -1 - (distance_to_target / self.grid_size)  # Penalize moving away from target
        
#         self.state = new_state
#         done = (new_state == self.target)

#         return np.array(self.state, dtype=np.float32), reward, done

# # ✅ Define DQN Model
# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# # ✅ Define DQN Agent
# class DQNAgent:
#     def __init__(self, env, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
#         self.env = env
#         self.memory = deque(maxlen=10000)
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon
#         self.model = DQN(2, len(env.actions)).to(device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
#     def choose_action(self, state):
#         if random.uniform(0, 1) < self.epsilon:
#             return random.choice(range(len(self.env.actions)))
#         else:
#             state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#             with torch.no_grad():
#                 action_values = self.model(state_tensor)
#             return torch.argmax(action_values).item()
    
#     def store_experience(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
    
#     def train(self, batch_size=32):
#         if len(self.memory) < batch_size:
#             return
        
#         batch = random.sample(self.memory, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.tensor(states, dtype=torch.float32, device=device)
#         actions = torch.tensor(actions, dtype=torch.long, device=device)
#         rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
#         next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
#         dones = torch.tensor(dones, dtype=torch.float32, device=device)

#         current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_q_values = self.model(next_states).max(1)[0].detach()
#         target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

#         loss = F.mse_loss(current_q_values, target_q_values)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# # ✅ Train DQN Agent with Dataset-Based Rewards
# def train_agent(env, agent, episodes):
#     rewards = []
#     for episode in range(episodes):
#         state = env.reset()
#         total_reward = 0
#         done = False

#         while not done:
#             action_idx = agent.choose_action(state)
#             action = env.actions[action_idx]
#             next_state, reward, done = env.step(action)
#             agent.store_experience(state, action_idx, reward, next_state, done)
#             agent.train(batch_size=32)

#             state = next_state
#             total_reward += reward

#         rewards.append(total_reward)
#         if episode % 100 == 0:
#             print(f"Episode {episode}: Total Reward = {total_reward}")

#     return rewards

# # ✅ Run Training
# target_position = (9, 9)
# env = UAVEnvironment(grid_size, target_position, obstacle_positions)
# agent = DQNAgent(env)

# episodes = 1000  # Can be increased for better learning
# training_rewards = train_agent(env, agent, episodes)

# # ✅ Plot Training Performance
# plt.plot(training_rewards)
# plt.xlabel("Episodes")
# plt.ylabel("Total Reward")
# plt.title("Training Performance - Deep Q-Learning with Dataset-Based Rewards")
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Dataset
csv_file_path = "D:\\building projects\\uav_project\\UAV_full_data.csv"  
data = pd.read_csv(csv_file_path)

# ✅ Extract obstacles from dataset
obstacle_positions = {}
grid_size = 10  

for _, row in data.iterrows():
    try:
        x, y = int(row['Start Frame']), int(row['End Frame'])
        severity = row.get("Obstacle Severity", 10)  # Default penalty if missing
        obstacle_positions[(x % grid_size, y % grid_size)] = severity
    except ValueError:
        continue

if not obstacle_positions:
    print("⚠️ Warning: No obstacles detected in dataset!")

# ✅ Define UAV Environment
class UAVEnvironment:
    def __init__(self, grid_size, target, obstacles):
        self.grid_size = grid_size
        self.state = (0, 0)  
        self.target = target  
        self.obstacles = obstacles  
        self.actions = ["up", "down", "left", "right"]

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        x, y = self.state
        if action == "up":
            x = max(x - 1, 0)
        elif action == "down":
            x = min(x + 1, self.grid_size - 1)
        elif action == "left":
            y = max(y - 1, 0)
        elif action == "right":
            y = min(y + 1, self.grid_size - 1)

        new_state = (x, y)
        distance_to_target = np.linalg.norm(np.array(new_state) - np.array(self.target))

        # ✅ **Dataset-Based Reward Calculation**
        if new_state in self.obstacles:
            reward = -self.obstacles[new_state]  # Penalty based on obstacle severity
        elif new_state == self.target:
            reward = 100  
        else:
            reward = -1 - (distance_to_target / self.grid_size)  # Penalize moving away from target
        
        self.state = new_state
        done = (new_state == self.target)

        return np.array(self.state, dtype=np.float32), reward, done

# ✅ Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Define DQN Agent
class DQNAgent:
    def __init__(self, env, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = DQN(2, len(env.actions)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.env.actions)))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ✅ Train DQN Agent Until a Good Reward is Achieved
def train_agent(env, agent, max_episodes=5000, reward_threshold=80):
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.choose_action(state)
            action = env.actions[action_idx]
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action_idx, reward, next_state, done)
            agent.train(batch_size=32)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

        # ✅ Stop training early if reward threshold is met
        if np.mean(rewards[-50:]) >= reward_threshold:
            print(f"✅ Training stopped at episode {episode} (Average reward: {np.mean(rewards[-50:])})")
            break

    return rewards

# ✅ Run Training
target_position = (9, 9)
env = UAVEnvironment(grid_size, target_position, obstacle_positions)
agent = DQNAgent(env)

training_rewards = train_agent(env, agent)

# ✅ Plot Training Performance
plt.plot(training_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Performance - UAV DQN with Dataset-Based Rewards")
plt.show()
