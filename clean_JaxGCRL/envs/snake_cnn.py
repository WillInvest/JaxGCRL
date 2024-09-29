import numpy as np
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
    """Custom Snake Environment that follows the Gym API and uses a grid representation."""
    
    def __init__(self, w=640, h=480, render_mode=False):
        super(SnakeEnv, self).__init__()
        self.w = w
        self.h = h
        self.grid_size = 20  # Assuming each block is 20x20 pixels
        self.grid_width = self.w // self.grid_size
        self.grid_height = self.h // self.grid_size
        self.render_mode = render_mode
        
        # Set observation space as a grid
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.grid_height, self.grid_width, 1), dtype=np.float32)
        
        # Action space: [straight, right, left]
        self.action_space = spaces.Discrete(3)

        self.display = pygame.Surface((self.w, self.h)) if render_mode else None
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None, options=None):
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if self.render_mode:
            self.video_writer = cv2.VideoWriter(f'/home/shiftpub/JaxGCRL/clean_JaxGCRL/video/snake_game.avi', 
                                                cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.w, self.h))
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - self.grid_size, self.head.y), Point(self.head.x - 2 * self.grid_size, self.head.y)]
        self.food = None
        self.score = 0
        self.steps = 0
        self._place_food()
        return self.get_state(), {}


    def _place_food(self):
        x = random.randint(0, (self.w - self.grid_size) // self.grid_size) * self.grid_size
        y = random.randint(0, (self.h - self.grid_size) // self.grid_size) * self.grid_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        """Represent the environment as a grid where each cell represents walls, snake, or food."""
        # Initialize the grid with 0 for empty cells
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Set the boundaries (walls) to -1
        grid[0, :] = -1.0  # Top wall
        grid[-1, :] = -1.0  # Bottom wall
        grid[:, 0] = -1.0  # Left wall
        grid[:, -1] = -1.0  # Right wall

        # Mark the snake's body with -1
        for segment in self.snake[1:]:  # Exclude the head for now
            grid[int(segment.y // self.grid_size)-1, int(segment.x // self.grid_size)-1] = -1.0  # Snake's body

        # Mark the snake's head with 1
        head_x = int(self.head.x // self.grid_size)
        head_y = int(self.head.y // self.grid_size)
        grid[head_y-1, head_x-1] = 1.0  # Snake's head
        
        # Mark the food with 2
        food_x = int(self.food.x // self.grid_size)
        food_y = int(self.food.y // self.grid_size)
        grid[food_y-1, food_x-1] = 2.0  # Food

        # Return the grid as the observation (with an added channel dimension)
        return grid[:, :, np.newaxis]  # Add a channel dimension to the grid


    def step(self, action):
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        done = False

        if self.is_collision():
            done = True
            reward = -10
            return self.get_state(), reward, done, False, self._get_info()

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # If no food is eaten, remove the tail of the snake
            
        # Update UI and clock speed
        if self.render_mode:
            self._update_ui()
            self.clock.tick(SPEED)

        return self.get_state(), reward, done, False, self._get_info()

    def _move(self, action):
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

        # Update the snake's head position
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += self.grid_size
        elif self.direction == Direction.LEFT:
            x -= self.grid_size
        elif self.direction == Direction.DOWN:
            y += self.grid_size
        elif self.direction == Direction.UP:
            y -= self.grid_size

        self.head = Point(x, y)
        
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

    def _get_info(self):
        return {'score': self.score, 'steps': self.steps}

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
                f"Score: {self.score}",  # The text to display
                text_position,  # Position of the text (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                1,  # Font scale
                (255, 255, 255),  # Font color (white)
                2,  # Thickness
                cv2.LINE_AA  # Line type for anti-aliased text
            )

            # Write the frame with the text to the video
            self.video_writer.write(frame)

if __name__ == "__main__":
    env = SnakeEnv(render_mode=True)
    done = False
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info, _ = env.step(action)    
   