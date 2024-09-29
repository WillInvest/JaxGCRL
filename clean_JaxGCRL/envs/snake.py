import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium import spaces
import numpy as np
import cv2
import pygame
import random
from enum import Enum
from collections import namedtuple
from datetime import datetime
import os

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeEnv(gym.Env):
    """Custom Snake Environment that follows the Gym API"""

    def __init__(self, w=640, h=480, render_mode=False):
        super(SnakeEnv, self).__init__()
        self.w = w
        self.h = h
        self.render_mode = render_mode  # Add a render mode flag
        self.display = pygame.Surface((self.w, self.h))  # Create a surface for headless rendering
        self.clock = pygame.time.Clock()
        self.reset_times = 0
        self.max_length = 101
        # Observation space: snake's head position, food position, and direction (continuous space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10+100*2,), dtype=np.float32)

        # Action space: [straight, right, left]
        self.action_space = spaces.Discrete(3)
    


    def reset(self, seed=0):
        self.reset_times += 1
        if self.render_mode:
            self.video_writer = cv2.VideoWriter(f'/home/shiftpub/JaxGCRL/clean_JaxGCRL/video/snake_game_{self.reset_times}.avi', 
                                                cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.w, self.h))
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.steps = 0  # Number of steps taken
        self._place_food()
        self.target_distance = np.linalg.norm([self.head.x - self.food.x, self.head.y - self.food.y])
        self.frame_iteration = 0

        # Return initial observation
        return np.array(self.get_state(), dtype=np.float32), {}

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        self.frame_iteration += 1
        self.steps += 1  # Increment steps

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        current_distance = np.linalg.norm([self.head.x - self.food.x, self.head.y - self.food.y])
        reward = self.target_distance - current_distance
        self.target_distance = current_distance

        done = False
        if self.is_collision() or self.frame_iteration > 300:
            done = True
            if self.score > 0:
                print(f"Reset Times: {self.reset_times} | Game Over! Score: {self.score}")
            reward = -100  # Penalty for game over
            return np.array(self.get_state(), dtype=np.float32), reward, done, False, self._get_info()

        if self.head == self.food:
            self.score += 1
            reward += 10  # Reward for eating food
            self.frame_iteration = 0
            self._place_food()
            self.target_distance = np.linalg.norm([self.head.x - self.food.x, self.head.y - self.food.y])
            if len(self.snake) >= 101:
                done = True
                print(f"Reset Times: {self.reset_times} | Game Over! Score: {self.score}")
                return np.array(self.get_state(), dtype=np.float32), reward, done, False, self._get_info()
        else:
            self.snake.pop()

        # Update UI and clock speed
        self._update_ui()
        self.clock.tick(SPEED)

        return np.array(self.get_state(), dtype=np.float32), reward, done, False, self._get_info()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx]  # no change
        elif action == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # action == 2
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        # Move the snake's head
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):
        if self.render_mode:  # Only update the UI when render_mode is True
            self.display.fill(BLACK)
            
            # Draw the snake
            for pt in self.snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
            # Draw the food
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

            # Capture the frame after rendering the display
            frame = pygame.surfarray.pixels3d(self.display)
            frame = np.rot90(frame)  # Keep rotation for proper video playback
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            
            # Get the frame height to position the text at the bottom
            frame_height = frame.shape[0]
            text_position = (10, frame_height - 10)  # Position 10px from the left and bottom
            
            # Render the text on the rotated frame using OpenCV
            frame = cv2.putText(
                frame,  # The frame to draw the text on
                f"Reset: {self.reset_times} | Score: {self.score}",  # The text to display
                text_position,  # Position of the text (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                1,  # Font scale
                (255, 255, 255),  # Font color (white)
                2,  # Thickness
                cv2.LINE_AA  # Line type for anti-aliased text
            )

            # Write the frame with the text to the video
            self.video_writer.write(frame)


    def _get_info(self):
        """Returns additional info for logging metrics"""
        distance_to_food = np.linalg.norm([self.head.x - self.food.x, self.head.y - self.food.y])
        return {
            'score': self.score,
            'steps': self.steps,
            'distance_to_food': distance_to_food,
        }

    # def get_state(self):
    #     # Normalize the snake's head and food coordinates by dividing by the width (w) and height (h) of the environment
    #     return([
    #         self.head.x / self.w,  # Normalize head x-coordinate
    #         self.head.y / self.h,  # Normalize head y-coordinate
    #         self.food.x / self.w,  # Normalize food x-coordinate
    #         self.food.y / self.h,  # Normalize food y-coordinate
    #         (self.w - self.head.x) / self.w,  # Relative x distance to wall
    #         (self.h - self.head.y) / self.h,  # Relative y distance to wall
    #         int(self.direction == Direction.RIGHT),
    #         int(self.direction == Direction.LEFT),
    #         int(self.direction == Direction.UP),
    #         int(self.direction == Direction.DOWN)
    #     ])
    
    def get_state(self):
        """Represent the environment as a grid where each cell represents the wall, snake, or food."""
        # Initialize the grid with -1 for walls (outer boundary)
        grid = np.full((self.grid_height, self.grid_width), -1.0, dtype=np.float32)
    
        # Mark the snake's body with 0
        for segment in self.snake:
            grid[int(segment.y // self.grid_size), int(segment.x // self.grid_size)] = 0.0  # Snake's body
    
        # Mark the food with 1
        grid[int(self.food.y // self.grid_size), int(self.food.x // self.grid_size)] = 1.0  # Food

        # Return the grid as the observation (with an added channel dimension)
        return grid[:, :, np.newaxis]  # Add a channel dimension to the grid


     


    def render(self, mode='human'):
        # You can add rendering logic if you want to visualize the environment
        pass

    def close(self):
        if self.render_mode:
            self.video_writer.release()
        pygame.quit()

    def __del__(self):
        self.close()

# Testing the environment
if __name__ == '__main__':
    env = SnakeEnv()
    obs, _ = env.reset()
    print(f"obs: {obs}")
    action = env.action_space.sample()
    print(f"action: {action}")
