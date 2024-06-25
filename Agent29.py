import gym
from gym import spaces
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# Custom environment definition
class CustomGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomGridEnv, self).__init__()
        
        # Define action space: [linear velocity, angular velocity]
        self.action_space = spaces.Box(low=np.array([-1.0, -np.pi]), high=np.array([1.0, np.pi]), dtype=np.float32)
        
        # Define observation space: [agent_x, agent_y, orientation, target_x, target_y]
        self.observation_space = spaces.Box(low=np.array([-160.0, -160.0, -np.pi, -160.0, -160.0]), high=np.array([160.0, 160.0, np.pi, 160.0, 160.0]), dtype=np.float32)
        
        # Initial state
        self.state = None
        self.target = None
        self.max_steps = 200
        self.current_step = 0
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.screen_size = 320
        self.scale = self.screen_size / 320
        
        self.reset()

    def reset(self):
        # Initialize the agent's state and the target's position
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Agent starts at the center facing right
        self.target = np.random.uniform(low=-160.0, high=160.0, size=(2,)).astype(np.float32)
        
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Return the current observation
        obs = np.array([self.state[0], self.state[1], self.state[2], self.target[0], self.target[1]], dtype=np.float32)
        return obs

    def step(self, action):
        linear_vel, angular_vel = action
        self.current_step += 1

        # Update the agent's orientation
        self.state[2] += angular_vel
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle between -pi and pi

        # Calculate the change in position
        delta_x = linear_vel * np.cos(self.state[2])
        delta_y = linear_vel * np.sin(self.state[2])

        # Update the agent's position
        self.state[0] += delta_x
        self.state[1] += delta_y

        # Calculate distance to the target
        distance_to_target = np.linalg.norm(self.target - self.state[:2])

        # Reward function
        reward = -distance_to_target

        # Check if the agent has reached the target
        done = distance_to_target < 5.0  # Consider reaching target if within 5 pixels
        if done:
            reward += 100.0  # Bonus for reaching the target
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Custom Grid Environment')
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((255, 255, 255))  # Clear the screen with white
        self._draw_checkered_background()

        # Draw target as a star
        target_pos = (int(self.screen_size / 2 + self.target[0] * self.scale), int(self.screen_size / 2 + self.target[1] * self.scale))
        pygame.draw.polygon(self.screen, (255, 0, 0), self._create_star(target_pos, 10, 5))

        # Draw agent as an arrow
        agent_pos = (int(self.screen_size / 2 + self.state[0] * self.scale), int(self.screen_size / 2 + self.state[1] * self.scale))
        self._draw_arrow(agent_pos, self.state[2])

        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS for smoother rendering

    def _draw_checkered_background(self):
        # Function to draw a checkered background
        tile_size = 40  # Larger tile size for better visibility
        colors = [(200, 200, 200), (255, 255, 255)]

        for y in range(0, self.screen_size, tile_size):
            for x in range(0, self.screen_size, tile_size):
                rect = pygame.Rect(x, y, tile_size, tile_size)
                color = colors[((x // tile_size) + (y // tile_size)) % 2]
                pygame.draw.rect(self.screen, color, rect)

    def _create_star(self, position, size, num_points):
        # Function to create a star shape
        points = []
        angle = np.pi / num_points
        for i in range(2 * num_points):
            r = size if i % 2 == 0 else size / 2
            x = position[0] + r * np.cos(i * angle)
            y = position[1] + r * np.sin(i * angle)
            points.append((x, y))
        return points

    def _draw_arrow(self, position, orientation):
        # Function to draw an arrow shape
        length = 20  # Length of the arrow
        width = 10   # Width of the arrow

        # Define points for the arrow shape
        points = [
            (position[0] + length * np.cos(orientation), position[1] + length * np.sin(orientation)),  # Tip of the arrow
            (position[0] - length * np.cos(orientation) + width * np.cos(orientation + np.pi / 2), position[1] - length * np.sin(orientation) + width * np.sin(orientation + np.pi / 2)),  # Left tail
            (position[0] - length * np.cos(orientation) - width * np.cos(orientation + np.pi / 2), position[1] - length * np.sin(orientation) - width * np.sin(orientation + np.pi / 2))   # Right tail
        ]
        pygame.draw.polygon(self.screen, (0, 0, 255), points)

    def close(self):
        if self.screen is not None:
            pygame.quit()

# DQN implementation using PyTorch
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

discrete_actions = [
    (0.5, 0.0),
    (0.5, 0.1),
    (0.5, -0.1),
    (1.0, 0.0),
    (1.0, 0.1),
    (1.0, -0.1),
    (1.5, 0.0),
    (1.5, 0.1),
    (1.5, -0.1),
    (-0.5, 0.0),
    (-0.5, 0.1),
    (-0.5, -0.1),
    (-1.0, 0.0),
    (-1.0, 0.1),
    (-1.0, -0.1),
    (-1.5, 0.0),
    (-1.5, 0.1),
    (-1.5, -0.1)
]

def epsilon_greedy_policy(state, epsilon):
    if random.random() < epsilon:
        action_index = random.choice(range(len(discrete_actions)))
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state)
        action_index = q_values.max(1)[1].item()
    return discrete_actions[action_index], action_index

def update_model():
    if len(replay_buffer) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    q_values = model(states)
    next_q_values = model(next_states)
    next_q_state_values = target_model(next_states)
    
    q_value = q_values.gather(1, actions).squeeze(1)
    next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    
    loss = criterion(q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = CustomGridEnv()
model = DQN(env.observation_space.shape[0], len(discrete_actions)).to(device)
target_model = DQN(env.observation_space.shape[0], len(discrete_actions)).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(10000)

num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 300
target_update = 10

all_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    prev_distance = np.linalg.norm(env.target - env.state[:2])
    
    for t in range(env.max_steps):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        action, action_index = epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        current_distance = np.linalg.norm(env.target - env.state[:2])
        
        # Reward based on distance reduction
        reward = prev_distance - current_distance
        prev_distance = current_distance

        if done:
            reward += 100.0
        
        print(f"Episode: {episode}, Step: {t}, Action: {action}, State: {state}, Reward: {reward}, Distance to target: {current_distance}")  # Debug statement

        replay_buffer.push(state, action_index, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        update_model()
        
        # Render the environment at each step
        env.render()
        
        if done:
            break
    
    all_rewards.append(episode_reward)
    
    if episode % target_update == 0:
        target_model.load_state_dict(model.state_dict())
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")
    
    # Short delay to allow Pygame window to refresh
    time.sleep(0.1)

env.close()

plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training Progress')
plt.show()
