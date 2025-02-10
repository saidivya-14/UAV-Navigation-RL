import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load UAV dataset
csv_file_path = "D:\\building projects\\uav_project\\UAV_full_data.csv"  # Ensure the file path is correct
data = pd.read_csv(csv_file_path)

# Define the environment
class UAVEnvironment:
    def __init__(self, grid_size, target):
        self.grid_size = grid_size
        self.state = (0, 0)  # Initial position
        self.target = target  # Target position
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

        self.state = (x, y)

        # Calculate reward
        if self.state == self.target:
            return self.state, 10, True  # Reached the target
        else:
            return self.state, -1, False  # Penalize each step

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.env = env
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(len(self.env.actions)))
        else:
            x, y = state
            return np.argmax(self.q_table[x, y, :])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        td_target = reward + self.discount_factor * np.max(self.q_table[next_x, next_y, :])
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.learning_rate * td_error

    def decay_exploration(self, decay_rate):
        self.exploration_rate *= decay_rate

# Train the agent
def train_agent(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action_idx = agent.choose_action(state)
            action = env.actions[action_idx]
            next_state, reward, done = env.step(action)

            agent.learn(state, action_idx, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_exploration(0.99)  # Decay exploration rate
        rewards.append(total_reward)

    return rewards

# Initialize environment and agent
grid_size = 5
target_position = (4, 4)

env = UAVEnvironment(grid_size, target_position)
agent = QLearningAgent(env)

# Train the agent
training_rewards = train_agent(env, agent, 1000)

# Plot training rewards
plt.plot(training_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Performance")
plt.show()
