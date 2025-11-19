import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.distributions import Categorical
import imageio
from PIL import Image
import matplotlib.pyplot as plt
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

def testing(agent, test_pairs, agent_type, num_episodes=None, env_config=None, save_gifs=True, gif_folder="TEST_GIFS"):
    """
    Unified testing function for both DQN and PPO agents
    
    Args:
        agent: The trained agent (DQN or PPO)
        test_pairs: List of (image_path, mask_path) tuples
        agent_type: Either "dqn" or "ppo"
        num_episodes: Number of episodes to test (default: all test pairs)
        env_config: Environment configuration dictionary
        save_gifs: Whether to save GIFs of episodes
        gif_folder: Folder to save GIFs
    
    Returns:
        Dictionary with test results including success rate and action distributions
    """
    if num_episodes is None:
        num_episodes = len(test_pairs)
    
    # Create GIF folder if needed
    if save_gifs and not os.path.exists(gif_folder):
        os.makedirs(gif_folder)
    
    # Set model to evaluation mode
    if agent_type.lower() == "dqn":
        agent.dnnetwork.eval()
    elif agent_type.lower() == "ppo":
        agent.model.eval()
    
    results = {
        'success_rate': [],
        'final_position_accuracy': [],
        'average_reward': [],
        'steps_to_find_tumor': [],
        'total_tumor_reward': [],
        'tumor_sizes_pixels': [],
        'tumor_sizes_percentage': [],
        'episode_details': []
    }
    
    grid_size = env_config.get('grid_size', 4)
    rewards = env_config.get('rewards', [5.0, -1.0, -0.2])
    action_space = env_config.get('action_space', None)
    
    for i in range(min(num_episodes, len(test_pairs))):
        img_path, mask_path = test_pairs[i]
        
        # Create environment
        if hasattr(agent, 'env_class'):
            env = agent.env_class(img_path, mask_path, grid_size=grid_size, rewards=rewards, action_space=action_space)
        else:
            env = Glioblastoma(img_path, mask_path, grid_size=grid_size, rewards=rewards, action_space=action_space)
        
        state, _ = env.reset()
        total_reward = 0
        found_tumor = False
        tumor_positions_visited = set()
        steps_to_find = env.max_steps
        tumor_rewards = 0
        
        # For action distribution tracking
        action_counts = np.zeros(env.action_space.n)
        
        # For GIF creation
        frames = []
        
        # Get tumor size information for this episode
        tumor_size_pixels = count_tumor_pixels(env)
        total_pixels = env.image.shape[0] * env.image.shape[1]
        tumor_size_percentage = (tumor_size_pixels / total_pixels) * 100
        
        results['tumor_sizes_pixels'].append(tumor_size_pixels)
        results['tumor_sizes_percentage'].append(tumor_size_percentage)
        
        for step in range(env.max_steps):
            with torch.no_grad():
                if agent_type.lower() == "dqn":
                    action = agent.dnnetwork.get_action(state, epsilon=0.00)
                    action_idx = action
                elif agent_type.lower() == "ppo":
                    action_probs, _ = agent.model(state)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    action_idx = action.item()
                elif agent_type.lower() == "reinforce":
                    action, _ = agent.policy.act(state)  # handles tensor conversion internally
                    action_idx = action

            
            action_counts[action_idx] += 1
            
            next_state, reward, terminated, truncated, _ = env.step(action_idx)
            state = next_state
            total_reward += reward
            
            # Track tumor-related metrics
            current_overlap = env.current_patch_overlap_with_lesion()
            if current_overlap > 0:
                tumor_positions_visited.add(tuple(env.agent_pos))
                if not found_tumor:
                    found_tumor = True
                    steps_to_find = step + 1
                
                # Count positive rewards (when on tumor)
                if reward > 0:
                    tumor_rewards += 1
            
            # Capture frame for GIF
            if save_gifs:
                frame = env.render(show=False)
                if frame is not None:
                    frames.append(frame)
            
            if terminated or truncated:
                break
        
        # Save GIF
        gif_path = None
        if save_gifs and frames:
            gif_path = os.path.join(gif_folder, f"episode_{i}_{os.path.basename(img_path).split('.')[0]}.gif")
            # Convert frames to PIL Images and save as GIF
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=500,  # milliseconds per frame
                loop=0
            )
            if i % 10 == 0:
                print(f"Saved GIF for episode {i} at {gif_path}")
        
        # Calculate metrics for this episode
        final_overlap = env.current_patch_overlap_with_lesion()
        
        # Success: ended on tumor region
        success = final_overlap > 0
        results['success_rate'].append(success)
        
        # Final position accuracy
        results['final_position_accuracy'].append(final_overlap > 0)
        
        # Average reward
        results['average_reward'].append(total_reward)
        
        # Steps to find tumor
        results['steps_to_find_tumor'].append(steps_to_find)
                
        # Total positive rewards from tumor
        results['total_tumor_reward'].append(tumor_rewards)
        
        # Store detailed episode information
        episode_detail = {
            'image_path': img_path,
            'success': success,
            'final_on_tumor': final_overlap > 0,
            'total_reward': total_reward,
            'steps_to_find_tumor': steps_to_find,
            'tumor_rewards': tumor_rewards,
            'tumor_size_pixels': tumor_size_pixels,
            'tumor_size_percentage': tumor_size_percentage,
            'action_distribution': action_counts / np.sum(action_counts),  # Normalized
            'action_counts_raw': action_counts,  # Keep raw counts for aggregation
            'gif_path': gif_path
        }
        results['episode_details'].append(episode_detail)
    
    # Calculate separate action distributions
    successful_episodes = [ep for ep in results['episode_details'] if ep['final_on_tumor']]
    unsuccessful_episodes = [ep for ep in results['episode_details'] if not ep['final_on_tumor']]
    
    action_dist_success = calculate_separate_action_distribution(successful_episodes)
    action_dist_failure = calculate_separate_action_distribution(unsuccessful_episodes)
    
    # Calculate overall metrics with new tumor size statistics
    overall_results = {
        'success_rate': np.mean(results['success_rate']),
        'average_reward': np.mean(results['average_reward']),
        'avg_steps_to_find_tumor': np.mean(results['steps_to_find_tumor']),
        'avg_tumor_rewards': np.mean(results['total_tumor_reward']),
        'biggest_tumor_pixels': np.max(results['tumor_sizes_pixels']),
        'smallest_tumor_pixels': np.min(results['tumor_sizes_pixels']),
        'biggest_tumor_percentage': np.max(results['tumor_sizes_percentage']),
        'smallest_tumor_percentage': np.min(results['tumor_sizes_percentage']),
        'avg_tumor_size_pixels': np.mean(results['tumor_sizes_pixels']),
        'avg_tumor_size_percentage': np.mean(results['tumor_sizes_percentage']),
        'action_distribution': calculate_overall_action_distribution(results['episode_details']),
        'action_distribution_success': action_dist_success,
        'action_distribution_failure': action_dist_failure,
        'episode_details': results['episode_details']
    }
    
    # Print summary
    print("\n" + "="*60)
    print(f"TEST RESULTS ({agent_type.upper()} Agent)")
    print("="*60)
    print(f"Success Rate: {overall_results['success_rate']*100:.2f}%")
    print(f"Average Episode Reward: {overall_results['average_reward']:.2f}")
    print(f"Average Steps to Find Tumor: {overall_results['avg_steps_to_find_tumor']:.2f}")
    print(f"Average Tumor Rewards per Episode: {overall_results['avg_tumor_rewards']:.2f}")
    print(f"Tumor Size Statistics:")
    print(f"  Biggest Tumor: {overall_results['biggest_tumor_pixels']:.0f} pixels ({overall_results['biggest_tumor_percentage']:.2f}%)")
    print(f"  Smallest Tumor: {overall_results['smallest_tumor_pixels']:.0f} pixels ({overall_results['smallest_tumor_percentage']:.2f}%)")
    print(f"  Average Tumor: {overall_results['avg_tumor_size_pixels']:.0f} pixels ({overall_results['avg_tumor_size_percentage']:.2f}%)")
    print(f"Overall Action Distribution: {overall_results['action_distribution']}")
    print(f"  Successful Episodes: {overall_results['action_distribution_success']}")
    print(f"  Unsuccessful Episodes: {overall_results['action_distribution_failure']}")
    
    # Print individual episode results
    print(f"\nDetailed Results for {len(results['episode_details'])} episodes:")
    print("-" * 80)
    for i, detail in enumerate(results['episode_details']):
        print(f"Episode {i}: {os.path.basename(detail['image_path'])}")
        print(f"  Success: {detail['success']}, Final on Tumor: {detail['final_on_tumor']}")
        print(f"  Total Reward: {detail['total_reward']:.2f}, Steps to Find: {detail['steps_to_find_tumor']}")
        print(f"  Tumor Size: {detail['tumor_size_pixels']} pixels ({detail['tumor_size_percentage']:.2f}%)")
        print(f"  Action Distribution: {detail['action_distribution']}")
        if detail['gif_path']:
            print(f"  GIF saved: {detail['gif_path']}")
        print()
    
    return overall_results

def count_tumor_pixels(env):
    """Count total number of tumor pixels in the mask"""
    if hasattr(env, 'mask'):
        return np.sum(env.mask > 0)
    elif hasattr(env, 'original_mask'):
        return np.sum(env.original_mask > 0)
    else:
        # Fallback: try to access the mask through available attributes
        try:
            mask = env.lesion_mask if hasattr(env, 'lesion_mask') else None
            if mask is not None:
                return np.sum(mask > 0)
        except:
            pass
    return 0

def calculate_overall_action_distribution(episode_details):
    """Calculate overall action distribution across all episodes"""
    total_actions = np.zeros_like(episode_details[0]['action_distribution'])
    
    for detail in episode_details:
        # Multiply by steps to get actual count, then normalize
        action_dist = detail['action_distribution']
        # Since action_distribution is already normalized per episode, we'll average them
        total_actions += action_dist
    
    # Normalize to get overall distribution
    overall_dist = total_actions / len(episode_details)
    return overall_dist

def calculate_separate_action_distribution(episode_list):
    """Calculate action distribution for a specific list of episodes"""
    if len(episode_list) == 0:
        return np.array([])  # Return empty array if no episodes
    
    total_actions = np.zeros_like(episode_list[0]['action_distribution'])
    
    for episode in episode_list:
        total_actions += episode['action_distribution']
    
    # Normalize to get distribution
    distribution = total_actions / len(episode_list)
    return distribution


#####################################################################################################

class Glioblastoma(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 
    # The metadata of the environment, e.g. {“render_modes”: [“rgb_array”, “human”], “render_fps”: 30}. 
    # For Jax or Torch, this can be indicated to users with “jax”=True or “torch”=True.

    def __init__(self, image_path, mask_path, grid_size=4, tumor_threshold=0.0001, rewards = [1.0, -2.0, -0.5], action_space=spaces.Discrete(3), max_steps=20, stop = True, render_mode="human"): # cosntructor with the brain image, the mask and a size
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
        
        self.stop = stop
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
        self.max_steps = max_steps  # like in the paper

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
        
        if self.stop == True:
            # check if previous position had tumor
            prev_overlap = self.current_patch_overlap_with_lesion()

            # ============================================================
            #                IMPLICIT STOP BEHAVIOR
            # ============================================================
            if prev_overlap > 0 and action == 0:
                reward = +8.0              # Reward for correctly stopping
                terminated = True
                obs = self._get_obs()
                return obs, reward, terminated, False, {}
            
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
        attempted_move_but_blocked = (action != 0) and (prev_pos == self.agent_pos)
        if attempted_move_but_blocked:
            #print("Out of bounds move attempted") # DEBUGGING
            return -1.0  # penalty for trying to move out of bounds
        
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
            # will not distinguish between moving on tumor or staying on tumor
            # since it returns, it will not execute the rest of the code
        else:
            if action == 0:  # stayed in place but no tumor.
                return self.rewards[1]
            else:
                return self.rewards[2]  # moved but no tumor

    def render(self, show=True):
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

        if show:
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

            ax.set_title(f"Agent at {self.agent_pos} | Step {self.current_step}/{self.max_steps}")
            ax.axis('off')
            plt.show()
            return None
        else: #just return without showing but draw the agent position
            rgb_array = (vis_img * 255).astype(np.uint8)
        
            # Draw grid lines directly on the array
            for i in range(1, self.grid_size):
                # Horizontal line
                y = i * self.block_size
                rgb_array[y-1:y+1, :] = [255, 255, 255]  # White line
                
                # Vertical line  
                x = i * self.block_size
                rgb_array[:, x-1:x+1] = [255, 255, 255]  # White line
            
            # Draw agent position as a yellow rectangle
            r0 = self.agent_pos[0] * self.block_size
            c0 = self.agent_pos[1] * self.block_size
            
            # Draw rectangle borders (yellow)
            rgb_array[r0:r0+2, c0:c0+self.block_size] = [255, 255, 0]  # Top border
            rgb_array[r0+self.block_size-2:r0+self.block_size, c0:c0+self.block_size] = [255, 255, 0]  # Bottom border
            rgb_array[r0:r0+self.block_size, c0:c0+2] = [255, 255, 0]  # Left border
            rgb_array[r0:r0+self.block_size, c0+self.block_size-2:c0+self.block_size] = [255, 255, 0]  # Right border
            
            # Add step counter text to the image
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(rgb_array)
            draw = ImageDraw.Draw(pil_img)
            
            # Use default font (you can also load a specific font)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw step counter in top-left corner
            step_text = f"Step: {self.current_step}/{self.max_steps}"
            draw.text((5, 5), step_text, fill=(255, 255, 0), font=font)  # Yellow text
            
            # Convert back to numpy array
            rgb_array = np.array(pil_img)
            return rgb_array

        
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

#####################################################################################################
# Glioblastoma2 has positional encodings
class GlioblastomaPositionalEncoding(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, image_path, mask_path, grid_size=4, tumor_threshold=0.0001, rewards = [1.0, -2.0, -0.5], action_space=spaces.Discrete(3), max_steps=20, stop = True, render_mode="human"): # cosntructor with the brain image, the mask and a size
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
        
        self.stop = stop
        self.render_mode = render_mode

        # UPDATED: Now 3 channels (image + 2 positional encodings)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.block_size, self.block_size),  # Changed from (60, 60) to (3, 60, 60)
            dtype=np.float32
        )

        self.agent_pos = [0, 0]
        self.current_step = 0
        self.max_steps = max_steps

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
        
        if self.stop == True:
            # check if previous position had tumor
            prev_overlap = self.current_patch_overlap_with_lesion()

            # ============================================================
            #                IMPLICIT STOP BEHAVIOR
            # ============================================================
            if prev_overlap > 0 and action == 0:
                reward = +8.0              # Reward for correctly stopping
                terminated = True
                obs = self._get_obs()
                return obs, reward, terminated, False, {}
        
        # Apply action (respect grid boundaries)
        if self.action_space == spaces.Discrete(3):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
        elif self.action_space == spaces.Discrete(5):
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0: # up
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[1] > 0: # left
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

    def render(self, show=True):
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

        if show:
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

            ax.set_title(f"Agent at {self.agent_pos} | Step {self.current_step}/{self.max_steps}")
            ax.axis('off')
            plt.show()
            return None
        else: #just return without showing but draw the agent position
            rgb_array = (vis_img * 255).astype(np.uint8)
        
            # Draw grid lines directly on the array
            for i in range(1, self.grid_size):
                # Horizontal line
                y = i * self.block_size
                rgb_array[y-1:y+1, :] = [255, 255, 255]  # White line
                
                # Vertical line  
                x = i * self.block_size
                rgb_array[:, x-1:x+1] = [255, 255, 255]  # White line
            
            # Draw agent position as a yellow rectangle
            r0 = self.agent_pos[0] * self.block_size
            c0 = self.agent_pos[1] * self.block_size
            
            # Draw rectangle borders (yellow)
            rgb_array[r0:r0+2, c0:c0+self.block_size] = [255, 255, 0]  # Top border
            rgb_array[r0+self.block_size-2:r0+self.block_size, c0:c0+self.block_size] = [255, 255, 0]  # Bottom border
            rgb_array[r0:r0+self.block_size, c0:c0+2] = [255, 255, 0]  # Left border
            rgb_array[r0:r0+self.block_size, c0+self.block_size-2:c0+self.block_size] = [255, 255, 0]  # Right border
            
            # Add step counter text to the image
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(rgb_array)
            draw = ImageDraw.Draw(pil_img)
            
            # Use default font (you can also load a specific font)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw step counter in top-left corner
            step_text = f"Step: {self.current_step}/{self.max_steps}"
            draw.text((5, 5), step_text, fill=(255, 255, 0), font=font)  # Yellow text
            
            # Convert back to numpy array
            rgb_array = np.array(pil_img)
            return rgb_array
        
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

