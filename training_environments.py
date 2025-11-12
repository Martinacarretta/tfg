import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from copy import deepcopy

import pandas as pd


SEED = 42

def prepare(mode = "train"):
    if mode == "train":
        base_dir = "/home/martina/codi2/4year/tfg/training_set_npy"
        csv_path = "/home/martina/codi2/4year/tfg/set_training.csv"
    else:
        base_dir = "/home/martina/codi2/4year/tfg/testing_set_npy"
        csv_path = "/home/martina/codi2/4year/tfg/set_testing.csv"

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Construct image and mask filenames
    df["image_path"] = df.apply(
        lambda row: os.path.join(base_dir, f"{row['Patient']:03d}_{row['SliceIndex']}.npy"), axis=1
    )
    df["mask_path"] = df.apply(
        lambda row: os.path.join(base_dir, f"{row['Patient']:03d}_{row['SliceIndex']}_mask.npy"), axis=1
    )

    # Sanity check (optional)
    pairs = [
        (img, mask)
        for img, mask in zip(df["image_path"], df["mask_path"])
        if os.path.exists(img) and os.path.exists(mask)
    ]

    print(f"✅ Found {len(pairs)} pairs out of {len(df)} listed in CSV.")
    return pairs

class Glioblastoma(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 
    # The metadata of the environment, e.g. {“render_modes”: [“rgb_array”, “human”], “render_fps”: 30}. 
    # For Jax or Torch, this can be indicated to users with “jax”=True or “torch”=True.

    def __init__(self, image_path, mask_path, grid_size=4, tumor_threshold=0.0001, rewards = [1.0, -2.0, -0.5], action_space=spaces.Discrete(3), render_mode="human"): # cosntructor with the brain image, the mask and a size
        super().__init__() # parent class
        
        self.image = np.load(image_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.uint8)
        
        img_min, img_max = self.image.min(), self.image.max()
        if img_max > 1.0:  # only normalize if not already in [0, 1]
            self.image = (self.image - img_min) / (img_max - img_min + 1e-8) #avoid division by 0

        self.grid_size = grid_size
        self.block_size = self.image.shape[0] // grid_size  # 240/4 = 60
        
        self.action_space = action_space
        self.tumor_threshold = tumor_threshold # 15% of the patch must be tumor to consider that the agent is inside the tumor region
        self.rewards = rewards  # [reward_on_tumor, reward_stay_no_tumor, reward_move_no_tumor]
        
        self.render_mode = render_mode

        # Observations: grayscale patch (normalized 0-1)
        # apparently Neural networks train better when inputs are scaled to small, 
        # consistent ranges rather than raw 0–255 values.
        self.observation_space = spaces.Box( # Supports continuous (and discrete) vectors or matrices
            low=0, high=1, # Data has been normalized
            shape=(self.block_size, self.block_size), # shape of the observation
            dtype=np.float32
        )

        self.agent_pos = [0, 0] # INITIAL POSITION AT TOP LEFT
        self.current_step = 0 # initialize counter
        self.max_steps = 20  # like in the paper

    def reset(self, seed=None, options=None): # new episode where we initialize the state. 
        super().reset(seed=seed) # parent
        
        # reset
        self.agent_pos = [0, 0]  # top-left corner
        self.current_step = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1

        prev_pos = self.agent_pos.copy() # for reward computation taking into consideration the transition changes
        
        # Apply action (respect grid boundaries)
        if self.action_space == spaces.Discrete(3):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1  # move down
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1  # move right
            # else, the agent doesn't move so the observation 
            # and reward will be calculated from the same position
            # no need to compute self.agent_pos
        elif self.action_space == spaces.Discrete(5):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1  # move down
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1  # move right
            elif action == 3 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1  # move up
            elif action == 4 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1  # move left
        elif self.action_space == spaces.Discrete(9):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1  # move down
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1  # move right
            elif action == 3 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1  # move up
            elif action == 4 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1  # move left
            elif action == 5 and self.agent_pos[0] < self.grid_size - 1 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[0] += 1  # move down-right
                self.agent_pos[1] += 1
            elif action == 6 and self.agent_pos[0] > 0 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[0] -= 1  # move up-right
                self.agent_pos[1] += 1
            elif action == 7 and self.agent_pos[0] < self.grid_size - 1 and self.agent_pos[1] > 0:
                self.agent_pos[0] += 1  # move down-left
                self.agent_pos[1] -= 1
            elif action == 8 and self.agent_pos[0] > 0 and self.agent_pos[1] > 0:
                self.agent_pos[0] -= 1  # move up-left
                self.agent_pos[1] -= 1
        
        reward = self._get_reward(action, prev_pos)
                
        obs = self._get_obs()

        # Episode ends
        terminated = self.current_step >= self.max_steps
        truncated = False  # we don’t need truncation here
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        r0 = self.agent_pos[0] * self.block_size # row start
        c0 = self.agent_pos[1] * self.block_size # col start
        
        patch = self.image[r0:r0+self.block_size, c0:c0+self.block_size].astype(np.float32)

        # if patch.max() == 0: # DEBUGGING
        #     print("Warning: extracted patch has max value 0 at position:", self.agent_pos)
        # else:
        #     print("Brain")
        return patch

    def _get_reward(self, action, prev_pos):        
        # look position of the agent in the mask
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        patch_mask = self.mask[r0:r0+self.block_size, c0:c0+self.block_size]
        
        # Now that i have the patch where i was and the patch where i am, i can check if there is tumor in any of them
        # tumor is labeled as 1 or 4 in the mask        
        # label 2 is edema
        
        # first get a count of the tumor pixels in the patch. 
        tumor_count_curr = np.sum(np.isin(patch_mask, [1, 4]))
        total = self.block_size * self.block_size # to compute the percentage
        # Determine if patch has more than self.tumor_threshold of tumor
        inside = (tumor_count_curr / total) >= self.tumor_threshold
        
        if inside:
            return self.rewards[0]  # reward for being on tumor or staying on tumor
        else:
            if action == 0 or prev_pos == self.agent_pos:  # stayed in place but no tumor. we are also taking into consideration that if the action was to move but we are at the edge of the grid, we also stay in place
                return self.rewards[1]
            else:
                return self.rewards[2]  # moved but no tumor

    def render(self):
        if self.render_mode != "human": # would be rgb_array or ansi
            return  # Only render in human mode

        # Create RGB visualization image
        # not necessary since it's grayscale, but i want to draw the mask and position
        vis_img = np.stack([self.image] * 3, axis=-1).astype(np.float32)

        # Overlay tumor mask in red [..., 0] 
        tumor_overlay = np.zeros_like(vis_img) # do all blank but here we have 3 channels, mask is 2D
        tumor_overlay[..., 0] = (self.mask > 0).astype(float) # red channel. set to float to avoid issues when blending in vis_img

        # transparency overlay (crec que es el mateix valor que tinc a l'altra notebook)
        alpha = 0.4
        vis_img = (1 - alpha) * vis_img + alpha * tumor_overlay

        # Plotting
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(vis_img, cmap='gray', origin='upper')

        # Draw grid lines
        # alpha for transparency again
        for i in range(1, self.grid_size):
            ax.axhline(i * self.block_size, color='white', lw=1, alpha=0.5)
            ax.axvline(i * self.block_size, color='white', lw=1, alpha=0.5)

        # Draw agent position
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        rect = patches.Rectangle(
            (c0, r0), # (x,y) bottom left corner
            self.block_size, # width
            self.block_size, # height
            linewidth=2,
            edgecolor='yellow',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_title(f"Agent at {self.agent_pos} | Step {self.current_step}")
        ax.axis('off')
        plt.show()
        
    def current_patch_overlap_with_lesion(self): # FALTAAA chat
        """ Returns the number of overlapping lesion pixels between the agent's current patch and the ground-truth mask. If > 0, the agent is correctly over the lesion (TP). """
        # get current agent patch boundaries
        row, col = self.agent_pos
        patch_h = self.block_size # not grid_size because grid_size is number of patches per side
        patch_w = self.block_size
        
        y0 = row * patch_h
        y1 = y0 + patch_h
        x0 = col * patch_w
        x1 = x0 + patch_w
        # extract mask region under current patch
        patch_mask = self.mask[y0:y1, x0:x1]
        # count how many pixels of lesion (nonzero)
        overlap = np.sum(patch_mask > 0)
        return overlap

# Glioblastoma2 has positional encodings
class Glioblastoma2(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, image_path, mask_path, grid_size=4, tumor_threshold=0.0001, rewards = [1.0, -2.0, -0.5], action_space=spaces.Discrete(3), render_mode="human"):
        super().__init__()
        
        self.image = np.load(image_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.uint8)
        
        img_min, img_max = self.image.min(), self.image.max()
        if img_max > 1.0:
            self.image = (self.image - img_min) / (img_max - img_min + 1e-8)

        self.grid_size = grid_size
        self.block_size = self.image.shape[0] // grid_size
        
        self.action_space = action_space
        self.tumor_threshold = tumor_threshold
        self.rewards = rewards
        
        self.render_mode = render_mode

        # UPDATED: Now 3 channels (image + 2 positional encodings)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.block_size, self.block_size),  # Changed from (60, 60) to (3, 60, 60)
            dtype=np.float32
        )

        self.agent_pos = [0, 0]
        self.current_step = 0
        self.max_steps = 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = [0, 0]
        self.current_step = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1

        prev_pos = self.agent_pos.copy()
        
        # Apply action (respect grid boundaries)
        if self.action_space == spaces.Discrete(3):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
        elif self.action_space == spaces.Discrete(5):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        elif self.action_space == spaces.Discrete(9):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            elif action == 5 and self.agent_pos[0] < self.grid_size - 1 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[0] += 1
                self.agent_pos[1] += 1
            elif action == 6 and self.agent_pos[0] > 0 and self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos[0] -= 1
                self.agent_pos[1] += 1
            elif action == 7 and self.agent_pos[0] < self.grid_size - 1 and self.agent_pos[1] > 0:
                self.agent_pos[0] += 1
                self.agent_pos[1] -= 1
            elif action == 8 and self.agent_pos[0] > 0 and self.agent_pos[1] > 0:
                self.agent_pos[0] -= 1
                self.agent_pos[1] -= 1
        
        reward = self._get_reward(action, prev_pos)
                
        obs = self._get_obs()

        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        UPDATED: Returns (3, 60, 60) tensor with:
        - Channel 0: Image patch
        - Channel 1: Normalized row position (0 to 1)
        - Channel 2: Normalized column position (0 to 1)
        """
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        
        # Extract image patch
        patch = self.image[r0:r0+self.block_size, c0:c0+self.block_size].astype(np.float32)
        
        # Create position encoding channels (normalized to [0, 1])
        pos_row = np.full_like(patch, self.agent_pos[0] / (self.grid_size - 1))
        pos_col = np.full_like(patch, self.agent_pos[1] / (self.grid_size - 1))
        
        # Stack into (3, H, W) format
        obs = np.stack([patch, pos_row, pos_col], axis=0)
        
        return obs

    def _get_reward(self, action, prev_pos):        
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        patch_mask = self.mask[r0:r0+self.block_size, c0:c0+self.block_size]
        
        tumor_count_curr = np.sum(np.isin(patch_mask, [1, 4]))
        total = self.block_size * self.block_size
        inside = (tumor_count_curr / total) >= self.tumor_threshold
        
        if inside:
            return self.rewards[0]
        else:
            if action == 0 or prev_pos == self.agent_pos:
                return self.rewards[1]
            else:
                return self.rewards[2]

    def render(self):
        if self.render_mode != "human":
            return

        vis_img = np.stack([self.image] * 3, axis=-1).astype(np.float32)

        tumor_overlay = np.zeros_like(vis_img)
        tumor_overlay[..., 0] = (self.mask > 0).astype(float)

        alpha = 0.4
        vis_img = (1 - alpha) * vis_img + alpha * tumor_overlay

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(vis_img, cmap='gray', origin='upper')

        for i in range(1, self.grid_size):
            ax.axhline(i * self.block_size, color='white', lw=1, alpha=0.5)
            ax.axvline(i * self.block_size, color='white', lw=1, alpha=0.5)

        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        rect = patches.Rectangle(
            (c0, r0),
            self.block_size,
            self.block_size,
            linewidth=2,
            edgecolor='yellow',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_title(f"Agent at {self.agent_pos} | Step {self.current_step}")
        ax.axis('off')
        plt.show()
        
    def current_patch_overlap_with_lesion(self):
        row, col = self.agent_pos
        patch_h = self.block_size
        patch_w = self.block_size
        
        y0 = row * patch_h
        y1 = y0 + patch_h
        x0 = col * patch_w
        x1 = x0 + patch_w
        
        patch_mask = self.mask[y0:y1, x0:x1]
        overlap = np.sum(patch_mask > 0)
        return overlap

