import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Load UAV dataset
csv_file_path = "D:\\building projects\\uav_project\\UAV_full_data.csv"  # Ensure the file path is correct
data = pd.read_csv(csv_file_path)

# Define obstacles from the dataset
obstacle_positions = set()
grid_size = 10  # Ensure grid size is set before using it
for _, row in data.iterrows():
    try:
        x, y = int(row['Start Frame']), int(row['End Frame'])
        obstacle_positions.add((x % grid_size, y % grid_size))  # Map to grid positions
    except ValueError:
        continue

# Ensure obstacles are correctly loaded
if not obstacle_positions:
    print("⚠️ Warning: No obstacles loaded from the dataset.")

# Define the environment
class UAVEnvironment:
    def __init__(self, grid_size, target, obstacles):
        self.grid_size = grid_size
        self.state = (0, 0)  # Initial position
        self.target = target  # Target position
        self.obstacles = obstacles  # Obstacles as a set of grid positions
        self.actions = ["up", "down", "left", "right"]

    def reset(self):
        self.state = (0, 0)  # Reset to the initial position
        return self.state

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

        # Check if the new state is an obstacle
        if new_state in self.obstacles:
            reward = -10  # Higher penalty for hitting an obstacle
            done = False
        else:
            self.state = new_state
            if self.state == self.target:
                reward = 50  # Higher reward for reaching the target
                done = True
            else:
                reward = -1  # Small penalty for each step
                done = False

        return self.state, reward, done

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.env = env
        self.memory = deque(maxlen=5000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = DQN(2, len(env.actions))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.env.actions)))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=16):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            output = self.model(state_tensor)[action]
            loss = F.mse_loss(output, torch.tensor(target, dtype=torch.float32))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Train the agent
def train_agent(env, agent, episodes, batch_size=16):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_idx = agent.choose_action(state)
            action = env.actions[action_idx]
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action_idx, reward, next_state, done)
            agent.train(batch_size)
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        rewards.append(total_reward)
    
    return rewards

episode = 0
target_position = (9, 9)

env = UAVEnvironment(grid_size, target_position, obstacle_positions)
agent = DQNAgent(env)

training_rewards = []
best_average_reward = -float('inf')  # Track best average reward
convergence_threshold = 10000  # Define a stopping condition
window_size = 10  # Check the last 10 episodes for improvement

while True:
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_idx = agent.choose_action(state)
        action = env.actions[action_idx]
        next_state, reward, done = env.step(action)

        agent.store_experience(state, action_idx, reward, next_state, done)
        agent.train(batch_size=16)

        state = next_state
        total_reward += reward

    training_rewards.append(total_reward)
    episode += 1

    # Print progress every 10 episodes
    if episode % 10 == 0:
        avg_reward = np.mean(training_rewards[-window_size:])
        print(f"✅ Episode {episode}: Average Reward = {avg_reward}")

        # Stop if the model is performing well
        if avg_reward >= convergence_threshold:
            print(f"🎯 Training complete at Episode {episode} (Reward = {avg_reward})!")
            break

plt.plot(training_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Performance - Deep Q-Learning")
plt.show()
