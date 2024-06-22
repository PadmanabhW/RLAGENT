import gym
from gym import spaces
import numpy as np
import pygame

class CustomGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomGridEnv, self).__init__()
        
        # Define action space: [linear velocity, angular velocity]
        self.action_space = spaces.Box(low=np.array([-1.0, -np.pi]), high=np.array([1.0, np.pi]), dtype=np.float32)
        
        # Define observation space: [target_x, target_y, orientation]
        self.observation_space = spaces.Box(low=np.array([-160.0, -160.0, -np.pi]), high=np.array([160.0, 160.0, np.pi]), dtype=np.float32)
        
        # Initial state
        self.state = None
        self.target = None
        self.max_steps = 1000
        self.current_step = 0
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.screen_size = 320
        self.scale = self.screen_size / 320

        # Teleoperation flag
        self.teleop = False
        
        self.reset()

    def reset(self):
        # Initialize the agent's state and the target's position
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Agent starts at the center facing right
        self.target = np.random.uniform(low=-160.0, high=160.0, size=(2,)).astype(np.float32)
        
        # Ensure the target is placed such that the agent has to turn
        while abs(np.arctan2(self.target[1], self.target[0])) < np.pi / 4:
            self.target = np.random.uniform(low=-160.0, high=160.0, size=(2,)).astype(np.float32)
        
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Return the current observation
        obs = np.array([self.target[0], self.target[1], self.state[2]], dtype=np.float32)
        return obs

    def step(self, action):
        linear_vel, angular_vel = action
        self.current_step += 1

        # Update the agent's orientation
        self.state[2] += angular_vel
        self.state[2] = np.mod(self.state[2], 2 * np.pi)  # Normalize angle between 0 and 2*pi

        # Calculate the change in position
        delta_x = linear_vel * np.cos(self.state[2])
        delta_y = linear_vel * np.sin(self.state[2])

        # Update the target's position relative to the agent
        self.target[0] -= delta_x
        self.target[1] -= delta_y

        # Random movement for the target if not teleoperated
        if not self.teleop:
            self._random_move_target()

        # Calculate distance to the target
        distance_to_target = np.linalg.norm(self.target)

        # Check if the agent has reached the target
        done = distance_to_target < 5.0  # Consider reaching target if within 5 pixels

        # Reward function
        reward = -distance_to_target
        if done:
            reward += 100.0  # Bonus for reaching the target
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def _random_move_target(self):
        # Randomly move the target
        move = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        self.target += move
        # Keep target within bounds
        self.target = np.clip(self.target, -160.0, 160.0)

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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.teleop = not self.teleop  # Toggle teleoperation mode
                if self.teleop:
                    self._teleop_target(event)

        self.screen.fill((255, 255, 255))  # Clear the screen with white
        self._draw_checkered_background()

        # Draw target as a star relative to the agent
        target_pos = (int(self.screen_size / 2 + self.target[0] * self.scale), int(self.screen_size / 2 + self.target[1] * self.scale))
        pygame.draw.polygon(self.screen, (255, 0, 0), self._create_star(target_pos, 10, 5))

        # Draw agent as an arrow in the center
        self._draw_arrow((self.screen_size // 2, self.screen_size // 2), self.state[2])

        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS for smoother rendering

    def _teleop_target(self, event):
        # Teleoperate the target with arrow keys
        if event.key == pygame.K_LEFT:
            self.target[0] -= 5
        elif event.key == pygame.K_RIGHT:
            self.target[0] += 5
        elif event.key == pygame.K_UP:
            self.target[1] -= 5
        elif event.key == pygame.K_DOWN:
            self.target[1] += 5
        # Keep target within bounds
        self.target = np.clip(self.target, -160.0, 160.0)

    def _draw_checkered_background(self):
        # Function to draw a checkered background
        tile_size = 40  # Larger tile size for better visibility
        colors = [(200, 200, 200), (255, 255, 255)]

        # Calculate the offset based on the agent's movement
        offset_x = int(self.target[0] * self.scale) % tile_size
        offset_y = int(self.target[1] * self.scale) % tile_size

        for y in range(-tile_size, self.screen_size + tile_size, tile_size):
            for x in range(-tile_size, self.screen_size + tile_size, tile_size):
                rect = pygame.Rect(x - offset_x, y - offset_y, tile_size, tile_size)
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

# Policy function to generate Vx and Wz based on the state
def policy(state):
    target_x, target_y, orientation = state

    # Calculate distance to the target
    distance_to_target = np.linalg.norm([target_x, target_y])
    
    # Calculate desired angle to the target
    desired_angle = np.arctan2(target_y, target_x)
    
    # Calculate angular velocity (difference between current orientation and desired angle)
    angular_velocity = desired_angle - orientation
    
    # Normalize angular velocity to be within [-pi, pi]
    angular_velocity = (angular_velocity + np.pi) % (2 * np.pi) - np.pi
    
    # Gradual turning
    angular_velocity = np.clip(angular_velocity, -0.1, 0.1)
    
    # Set linear velocity proportional to the distance to the target
    linear_velocity = 0.5  # Constant linear velocity for smoother movement

    return np.array([linear_velocity, angular_velocity], dtype=np.float32)

# Test the environment with the policy
if __name__ == "__main__":
    env = CustomGridEnv()
    obs = env.reset()
    for _ in range(10000):
        action = policy(obs)  # Use the policy to determine the action
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break
    env.close()
