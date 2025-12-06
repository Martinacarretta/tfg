import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.distributions import Categorical
from stable_baselines3 import PPO
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import sys



SEED = 42

def prepare(mode = "train", dataset = 0):
    if mode == "train":
        if dataset == 0:
            print
            base_dir = "/home/martina/codi2/4year/tfg/training_set_npy"
            csv_path = "/home/martina/codi2/4year/tfg/training_set.csv"
        elif dataset == 200:
            print("Using dataset of 200 samples for training.")
            base_dir = "/home/martina/codi2/4year/tfg/training_set_200_npy"
            csv_path = "/home/martina/codi2/4year/tfg/training_set_200.csv"
        else:
            print("Dataset not recognized, using default training set.")
            base_dir = "/home/martina/codi2/4year/tfg/training_set_npy"
            csv_path = "/home/martina/codi2/4year/tfg/training_set.csv"
    else:
        print("Preparing testing set.")
        base_dir = "/home/martina/codi2/4year/tfg/testing_set_npy"
        csv_path = "/home/martina/codi2/4year/tfg/testing_set.csv"

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
    Args:
        agent: The trained agent (DQN, PPO or REINFORCE)
        test_pairs: List of (image_path, mask_path) tuples
        agent_type: Either "dqn", "ppo" or "reinforce"
        num_episodes: Number of episodes to test (default: all test pairs)
        env_config: Environment configuration dictionary
        save_gifs: Whether to save GIFs of episodes
        gif_folder: Folder to save GIFs
    
    Returns:
        Dictionary with test results
    """
    if num_episodes is None:
        num_episodes = len(test_pairs)
    
    # Create GIF folder if needed
    if save_gifs and not os.path.exists(gif_folder):
        os.makedirs(gif_folder)
    
    # Set model to evaluation mode
    if agent_type.lower() == "dqn":
        agent.dnnetwork.eval()
    # elif agent_type.lower() == "ppo":
    #     agent.model.eval()
    
    results = {
        'hard_success': [], 'hard_failure': [],
        'timeout_success': [], 'timeout_failure': [],
        'average_reward': [],
        'steps_to_find_tumor': [], 'total_tumor_reward': [],
        'tumor_sizes_pixels': [], 'tumor_sizes_percentage': [],
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
            env = GlioblastomaPositionalEncoding(img_path, mask_path, grid_size=grid_size, rewards=rewards, action_space=action_space)
        
        state, _ = env.reset()
        terminated_by_stay = False
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
                    action, _states = agent.predict(state, deterministic=True)
                    action_idx = int(action)
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
                terminated_by_stay = (action_idx == 0)
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
        time_out = (not terminated_by_stay and (step+1) >= env.max_steps)
        final_overlap = env.current_patch_overlap_with_lesion()
        
        hard_success = terminated_by_stay and final_overlap > 0
        hard_failure = terminated_by_stay and final_overlap == 0
        timeout_success = time_out and final_overlap > 0
        timeout_failure = time_out and final_overlap == 0
    
        results['hard_success'].append(hard_success)
        results['hard_failure'].append(hard_failure)
        results['timeout_success'].append(timeout_success)
        results['timeout_failure'].append(timeout_failure)
        
        results['average_reward'].append(total_reward)
        results['steps_to_find_tumor'].append(steps_to_find)     
        results['total_tumor_reward'].append(tumor_rewards)
        
        # Store detailed episode information
        episode_detail = {
            'image_path': img_path,
            'terminated_by_stay': terminated_by_stay,
            'timed_out': time_out,
            'hard_success': hard_success,
            'hard_failure': hard_failure,
            'timeout_success': timeout_success,
            'timeout_failure': timeout_failure,
            'final_overlap': final_overlap > 0,
            'total_reward': total_reward,
            'steps_to_find_tumor': steps_to_find,
            'tumor_rewards': tumor_rewards,
            'tumor_size_pixels': tumor_size_pixels,
            'tumor_size_percentage': tumor_size_percentage,
            'action_distribution': action_counts / np.sum(action_counts),
            'action_counts_raw': action_counts,
            'gif_path': gif_path
        }
        results['episode_details'].append(episode_detail)
    
    # Calculate separate action distributions
    hard_success_eps = [ep for ep in results['episode_details'] if ep['hard_success']]
    hard_failure_eps = [ep for ep in results['episode_details'] if ep['hard_failure']]
    timeout_success_eps = [ep for ep in results['episode_details'] if ep['timeout_success']]
    timeout_failure_eps = [ep for ep in results['episode_details'] if ep['timeout_failure']]

    
    action_dist_hard_success = calculate_separate_action_distribution(hard_success_eps)
    action_dist_hard_failure = calculate_separate_action_distribution(hard_failure_eps)
    action_dist_timeout_success = calculate_separate_action_distribution(timeout_success_eps)
    action_dist_timeout_failure = calculate_separate_action_distribution(timeout_failure_eps)
    
    # Calculate overall metrics with new tumor size statistics
    overall_results = {
        'hard_success_rate': np.mean(results['hard_success']),
        'hard_failure_rate': np.mean(results['hard_failure']),
        'timeout_success_rate': np.mean(results['timeout_success']),
        'timeout_failure_rate': np.mean(results['timeout_failure']),
        
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
        'action_distribution_hard_success': action_dist_hard_success,
        'action_distribution_hard_failure': action_dist_hard_failure,
        'action_distribution_timeout_success': action_dist_timeout_success,
        'action_distribution_timeout_failure': action_dist_timeout_failure,
        'episode_details': results['episode_details']
    }
    
    # Print summary
    print("\n" + "="*60)
    print(f"TEST RESULTS ({agent_type.upper()} Agent)")
    print("="*60)
    print(f"✅Hard Success (correct STAY): {overall_results['hard_success_rate']*100:.2f}%")
    print(f"   ❌Hard Failure (wrong STAY): {overall_results['hard_failure_rate']*100:.2f}%")
    print(f"✔️Timeout Success (lucky): {overall_results['timeout_success_rate']*100:.2f}%")
    print(f"   ❌Timeout Failure: {overall_results['timeout_failure_rate']*100:.2f}%")

    print(f"Average Episode Reward: {overall_results['average_reward']:.2f}")
    print(f"Average Steps to Find Tumor: {overall_results['avg_steps_to_find_tumor']:.2f}")
    print(f"Average Tumor Rewards per Episode: {overall_results['avg_tumor_rewards']:.2f}")
    print(f"Tumor Size Statistics:")
    print(f"  Biggest Tumor: {overall_results['biggest_tumor_pixels']:.0f} pixels ({overall_results['biggest_tumor_percentage']:.2f}%)")
    print(f"  Smallest Tumor: {overall_results['smallest_tumor_pixels']:.0f} pixels ({overall_results['smallest_tumor_percentage']:.2f}%)")
    print(f"  Average Tumor: {overall_results['avg_tumor_size_pixels']:.0f} pixels ({overall_results['avg_tumor_size_percentage']:.2f}%)")
    print(f"Overall Action Distribution: {overall_results['action_distribution']}")
    print(f"  Hard Successful Episodes: {overall_results['action_distribution_hard_success']}")
    print(f"  Hard Unsuccessful Episodes: {overall_results['action_distribution_hard_failure']}")
    print(f"  Timeout Successful Episodes: {overall_results['action_distribution_timeout_success']}")
    print(f"  Timeout Unsuccessful Episodes: {overall_results['action_distribution_timeout_failure']}")
    
    # Print individual episode results
    print(f"\nDetailed Results for {len(results['episode_details'])} episodes:")
    print("-" * 80)
    for i, detail in enumerate(results['episode_details']):
        print(f"Episode {i}: {os.path.basename(detail['image_path'])}")
        print(f"  Hard Success: {detail['hard_success']}, Hard Failure: {detail['hard_failure']}")
        print(f"  Timeout Success: {detail['timeout_success']}, Timeout Failure: {detail['timeout_failure']}")
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


###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################
###############################################################################################################################################################################################################################################################################################################

class GlioblastomaPositionalEncoding(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, image_path, mask_path, grid_size=4, tumor_threshold=0.01, rewards = [10.0, -2.0, 2.5, -0.1], action_space=spaces.Discrete(3), max_steps=20, render_mode="human"): # cosntructor with the brain image, the mask and a size
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
        self.prev_pos = None
        self.prev_prev_pos = None
        self.current_step = 0
        if max_steps == 0:
            self.max_steps = sys.maxsize
        else:
            self.max_steps = max_steps
        
    def _random_shift(self):
        pad = 20
        H, W = self.image.shape

        while True:
            canvas = np.zeros((H + 2*pad, W + 2*pad), dtype=self.image.dtype)
            canvas_mask = np.zeros_like(canvas)

            # random offset
            y_off = np.random.randint(0, 2*pad+1)
            x_off = np.random.randint(0, 2*pad+1)

            # place original image
            canvas[y_off:y_off+H, x_off:x_off+W] = self.image
            canvas_mask[y_off:y_off+H, x_off:x_off+W] = self.mask

            # crop
            new_image = canvas[pad:pad+H, pad:pad+W]
            new_mask = canvas_mask[pad:pad+H, pad:pad+W]

            # check if new_mask still contains tumor
            if np.sum(new_mask > 0) > 0:
                self.image = new_image
                self.mask = new_mask
                return

    def reset(self, seed=None, options=None, force_on_target=False, start_on_zero=False):
        super().reset(seed=seed)
        
        self._random_shift()  # Apply random shift on reset

        if start_on_zero:
            self.agent_pos = [0, 0]
        else:
            if force_on_target: # start on tumor so it can see good reward if stay
                tumor_indices = np.where(self.mask > 0)
                # Pick a random pixel within the tumor
                idx = np.random.randint(len(tumor_indices[0]))
                one = tumor_indices[0][idx]
                two = tumor_indices[1][idx]
                self.agent_pos = [one // self.block_size, two // self.block_size]
            else:
                # Standard random start
                self.agent_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        
        self.current_step = 0
        self.prev_pos = None
        self.prev_prev_pos = None
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        prev_pos = self.agent_pos.copy()    # store position BEFORE applying action
                
        if action == 0: # END episode
            reward = self._get_reward(action, prev_pos)
            if reward == self.rewards[0]: # good stop
                terminated = True  
            else:    
                terminated = False
            obs = self._get_obs()
            return obs, reward, terminated, False, {}
        
        # Apply action (respect grid boundaries)
        if self.action_space.n == 3:
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
                
        elif self.action_space.n == 5:
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0: # up
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[1] > 0: # left
                self.agent_pos[1] -= 1
        
        reward = self._get_reward(action, prev_pos)
        
        terminated = self.current_step >= self.max_steps
        obs = self._get_obs()
        info = {}

        # track previous positions for oscillation detection
        self.prev_prev_pos = self.prev_pos.copy() if self.prev_pos is not None else None
        self.prev_pos = prev_pos.copy()     # store for next step

        return obs, reward, terminated, False, info

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
        # oscillation = agent returns to the previous position (A→B→A)
        if self.prev_pos is not None and self.agent_pos == self.prev_pos:
            return -1.0

  
        attempted_move_but_blocked = (action != 0) and (prev_pos == self.agent_pos)
        if attempted_move_but_blocked:
            #print("Out of bounds move attempted") # DEBUGGING
            return -0.3  # penalty for trying to move out of bounds
        
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
        inside = tumor_count_curr > 0 
        # inside = (tumor_count_curr / total) >= self.tumor_threshold

        if action == 0:   
            if inside:
                return self.rewards[0]
            else:
                return self.rewards[1]
        
        else: # movement
            if inside:
                return self.rewards[2]  # reward for moving into tumor
            else:
                # tumor_indices = np.where(self.mask > 0)
                # if len(tumor_indices[0]) == 0:
                #     # Fallback: If no tumor, set target to center of image, 
                #     ty, tx = self.image.shape[0] / 2, self.image.shape[1] / 2 
                # else:
                #     ty, tx = np.mean(tumor_indices, axis=1)

                # prev_dist = np.sqrt((prev_pos[1] - tx)**2 + (prev_pos[0] - ty)**2)
                # curr_dist = np.sqrt((self.agent_pos[1] - tx)**2 + (self.agent_pos[0] - ty)**2)                
                # #reward for moving closer, penalty for moving away
                # dist_delta = prev_dist - curr_dist
                # shaping_reward = dist_delta * 0.2
                # return self.rewards[3] + shaping_reward
                
                prev_overlap = self.current_patch_overlap_with_lesion(prev_pos)
                curr_overlap = self.current_patch_overlap_with_lesion(self.agent_pos)

                delta = curr_overlap - prev_overlap

                if delta > 0:
                    bonus = 0.2
                elif delta < 0:
                    bonus = -0.2
                else:
                    bonus = 0.0

                return self.rewards[3] + bonus


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
        
    def current_patch_overlap_with_lesion(self, pos=None): # FALTAAA chat
        """ Returns the number of overlapping lesion pixels between the agent's current patch and the ground-truth mask. If > 0, the agent is correctly over the lesion (TP). """
        if pos is None:
            row, col = self.agent_pos
        else:
            row, col = pos
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

