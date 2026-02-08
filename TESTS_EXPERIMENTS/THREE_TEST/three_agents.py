import os
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from PIL import Image, ImageDraw, ImageFont


from general import prepare
from env import GlioblastomaPositionalEncoding

from training_dqnpos import DQNPositionalEncoding
from training_agents import DQNAgent
from training_buffers import ReplayBuffer

import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
from torch.distributions import Categorical
import wandb

# ========================= PPO =========================

class DatasetWrapper(gym.Wrapper):
    def __init__(self, image_paths, mask_paths, **env_kwargs):
        # image_paths and mask_paths are lists of length N
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
            self.mask_paths = [mask_paths]
        else:
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            
        self.n = len(self.image_paths)
        self.env_kwargs = env_kwargs  # arguments to pass to inner env

        # TEMP env to inherit observation/action space
        tmp_env = GlioblastomaPositionalEncoding(self.image_paths[0], self.mask_paths[0], **env_kwargs)
        super().__init__(tmp_env)
        
        if self.env_kwargs.get('dataset') == 'polyp':
            # old_shape = self.observation_space.shape
            # new_shape = old_shape + (3,)
            # self.observation_space = spaces.Box(
            #     low=0,
            #     high=255,
            #     shape=new_shape,
            #     dtype=np.uint8
            # )
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(15, tmp_env.block_size, tmp_env.block_size),
                dtype=np.uint8
            )
    def __getattr__(self, name):
        """Forward all unknown attributes to the inner environment."""
        return getattr(self.env, name)
    def render(self, show=True):
        # Explicitly forward the 'show' argument to the internal environment
        return self.env.render(show=show)
    def reset(self, **kwargs):
        idx = np.random.randint(0, self.n)
        if hasattr(self, 'env'):
            self.env.close()
        self.env = GlioblastomaPositionalEncoding(
            self.image_paths[idx],
            self.mask_paths[idx],
            **self.env_kwargs
        )
        if 'start_on_zero' in kwargs:
            start_on_zero = kwargs.pop('start_on_zero')
        else:
            start_on_zero = np.random.random() < 0.5
            
        if self.env_kwargs.get('dataset') == 'polyp':
            start_on_zero = np.random.random() < 0.5
            obs, info = self.env.reset(**kwargs, start_on_zero=start_on_zero)
            obs = (obs * 255).astype(np.uint8)
            
            if obs.ndim == 4 and 3 in obs.shape:
                channel_axis = list(obs.shape).index(3)
                obs = np.moveaxis(obs, channel_axis, 1) # Move RGB to second position
                # This makes it (5, 3, 40, 40). Now reshape to (15, 40, 40)
                obs = obs.reshape(15, self.env.block_size, self.env.block_size)
            
            # CASE 2: 3D input (Slices, H, W) -> (5, 40, 40)
            elif obs.ndim == 3 and obs.shape[0] == 5:
                # We need 15 channels for the observation_space, but we only have 5.
                # We must repeat the channels to fill the 15-channel requirement.
                obs = np.repeat(obs, 3, axis=0) # (5, 40, 40) -> (15, 40, 40)

            return obs, info
        else:
            if np.random.random() < 0.5:
                return self.env.reset(**kwargs, start_on_zero=True)
            else:
                return self.env.reset(**kwargs, start_on_zero=False)

    def step(self, action):
        if self.env_kwargs.get('dataset') == 'polyp':
            obs, reward, terminated, truncated, info = self.env.step(action)  
            obs = (obs * 255).astype(np.uint8)

            if obs.ndim == 4 and 3 in obs.shape:
                channel_axis = list(obs.shape).index(3)
                obs = np.moveaxis(obs, channel_axis, 1) # Move RGB to second position
                # This makes it (5, 3, 40, 40). Now reshape to (15, 40, 40)
                obs = obs.reshape(15, self.env.block_size, self.env.block_size)
            
            # CASE 2: 3D input (Slices, H, W) -> (5, 40, 40)
            elif obs.ndim == 3 and obs.shape[0] == 5:
                # We need 15 channels for the observation_space, but we only have 5.
                # We must repeat the channels to fill the 15-channel requirement.
                obs = np.repeat(obs, 3, axis=0) # (5, 40, 40) -> (15, 40, 40)

            return obs, reward, terminated, truncated, info   
        else:     
            return self.env.step(action)

def make_env(grid_size, rewards, action_space, max_steps, dataset_name):
    def _init():
        env = DatasetWrapper(
            image_paths=image_paths,
            mask_paths=mask_paths,
            grid_size=grid_size,
            tumor_threshold=0.01,
            rewards=rewards,
            action_space=action_space,
            max_steps=max_steps,
            dataset=dataset_name
        )
        return Monitor(env)
    return _init


# ========================= REINFORCE =========================
class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim):
        """
        obs_shape: (C, H, W)
        action_dim: number of discrete actions
        """
        super().__init__()
        C, H, W = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute the flatten size instead of hardcoding 64*5*5
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: probs: (B, action_dim)
        """
        x = self.conv(x)
        x = x.flatten(1)
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def act(self, state, device):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        if state_t.ndim == 2:        # (H, W) -> (1, 1, H, W)
            state_t = state_t.unsqueeze(0).unsqueeze(0)
        elif state_t.ndim == 3:      # (C, H, W) -> (1, C, H, W)
            state_t = state_t.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state ndim={state_t.ndim}, shape={state_t.shape}")

        probs = self.forward(state_t)           # (1, action_dim)
        dist = Categorical(probs)
        action = dist.sample()                  # (1,)
        log_prob = dist.log_prob(action)  # (1,)
        dist_entropy = dist.entropy()    # (1,)
        return action.item(), log_prob.squeeze(0), dist_entropy.squeeze(0)  # scalar tensor

class REINFORCEAgent:
    def __init__(self, env_class, train_pairs, env_config,
                 gamma=0.99, lr=1e-4, save_path="reinforce_policy.pt"):

        self.env_class = env_class
        self.train_pairs = train_pairs
        self.env_config = env_config
        self.gamma = gamma
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Infer observation shape & action_dim from a sample env ---
        sample_img, sample_mask = train_pairs[0]
        sample_env = env_class(sample_img, sample_mask, **env_config)
        obs, _ = sample_env.reset()

        if obs.ndim == 2:
            C, H, W = 1, obs.shape[0], obs.shape[1]
        elif obs.ndim == 3:
            C, H, W = obs.shape
        else:
            raise ValueError(f"Unexpected obs ndim={obs.ndim}, shape={obs.shape}")

        obs_shape = (C, H, W)
        self.action_dim = env_config["action_space"].n

        # --- Policy network ---
        self.policy = CNNPolicy(obs_shape, self.action_dim).to(self.device)
        self.model = self.policy  # for compatibility with your testing() function
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.best_reward = -1e9

    def make_env(self, img_path, mask_path):
        return self.env_class(img_path, mask_path, **self.env_config)


    def compute_returns(self, rewards):
        """
        Compute discounted returns G_t for a single episode (no normalization here).
        """
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns  # plain Python list of floats

    def update_imitation(self, states, actions):
        """
        Phase 1: Behavioral Cloning (Imitation Learning)
        Uses standard Cross-Entropy to make the policy mimic the expert.
        """
        # Convert lists to tensors
        state_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)

        # Forward pass
        logits = self.policy.fc(self.policy.conv(state_t).flatten(1)) # Get raw logits
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, action_t)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def update_rl(self, log_probs, returns, entropies):
        """
        Phase 2: REINFORCE with Entropy Bonus
        """
        log_probs_t = torch.stack(log_probs)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        entropies_t = torch.stack(entropies)

        # Standardize returns to reduce variance
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy Gradient Loss + Entropy Bonus (0.01 weight)
        policy_loss = -(log_probs_t * returns_t).mean()
        entropy_loss = -0.01 * entropies_t.mean() 
        
        total_loss = policy_loss + entropy_loss
        
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optim.step()
        
        return total_loss.item()
    
    
    def expert_policy(self, env):
        # tumor centroid
        y, x = np.where(env.mask > 0)
        cy, cx = np.mean(y), np.mean(x)

        # current cell center
        gy = env.agent_pos[0] * env.block_size + env.block_size/2
        gx = env.agent_pos[1] * env.block_size + env.block_size/2

        # teacher picks the move that reduces distance the most
        moves = {
            1: (env.agent_pos[0] + 1, env.agent_pos[1]),  # down
            2: (env.agent_pos[0], env.agent_pos[1] + 1),  # right
            3: (env.agent_pos[0] - 1, env.agent_pos[1]),  # up
            4: (env.agent_pos[0], env.agent_pos[1] - 1),  # left
            0: (env.agent_pos[0], env.agent_pos[1])       # stay
        }

        best_action = 0
        best_dist = float("inf")

        for a, (ny, nx) in moves.items():
            if 0 <= ny < env.grid_size and 0 <= nx < env.grid_size:
                ncy = ny * env.block_size + env.block_size/2
                ncx = nx * env.block_size + env.block_size/2
                d = (ncx - cx)**2 + (ncy - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_action = a

        # when in tumor → STAY
        if env.current_patch_overlap_with_lesion() > 0:
            return 0

        return best_action

    def run_human_episode(self, img_path, mask_path):
        states, actions, rewards = [], [], []

        env = self.make_env(img_path, mask_path)
        state, _ = env.reset(force_on_target=False, start_on_zero=False) # random start
        done = False

        found = False
        steps_to_tumor = None
        step = 0

        while not done:
            action = self.expert_policy(env)
            next_state, reward, term, trunc, _ = env.step(action)

            if env.current_patch_overlap_with_lesion() > 0 and not found:
                found = True
                steps_to_tumor = step
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = term or trunc
            step += 1
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "found": found,
            "steps_to_tumor": steps_to_tumor,
            "episode_return": sum(rewards),
        }

    def run_episode_with_entropy(self, img_path, mask_path):
        env = self.make_env(img_path, mask_path)
        log_probs, rewards, entropies = [], [], []

        state, _ = env.reset()
        done = False
        while not done:
            action, log_prob, entropy = self.policy.act(state, self.device)
            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state
            done = terminated or truncated
        return log_probs, rewards, entropies

# ========================= Visualization =========================
def visualize_three_agents(img_path, mask_path, dqn_agent, ppo_agent, reinforce_agent, env_config):
    # Initialize the environment once to ensure same start position and tumor location
    env = GlioblastomaPositionalEncoding(img_path, mask_path, **env_config)
    
    # Track trajectories for each
    trajectories = {'dqn': [], 'ppo': [], 'reinforce': []}
    agents = {'dqn': dqn_agent, 'ppo': ppo_agent, 'reinforce': reinforce_agent}
    
    dqn_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
    ppo_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
    reinforce_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
    
    outcomes = {'dqn': "", 'ppo': "", 'reinforce': ""} # New dictionary

    for name, agent in agents.items():
        state, _ = env.reset(start_on_zero=True) # Fixed seed for same start
        done = False
        while not done:
            trajectories[name].append(env.agent_pos.copy())
                        
            # Logic to get action based on agent type
            with torch.no_grad():
                if name == "dqn":
                    action = agent.dnnetwork.get_action(state, epsilon=0.00)
                    action_idx = action
                elif name == "ppo":
                    # --- FIX START: Match DatasetWrapper preprocessing ---
                    ppo_state = state
                    if env_config.get('dataset') == 'polyp':
                        ppo_state = (ppo_state * 255).astype(np.uint8)
                        
                        # 2. Case 1: 4D input (5, 40, 40, 3) -> (15, 40, 40)
                        if ppo_state.ndim == 4 and 3 in ppo_state.shape:
                            try:
                                channel_axis = list(ppo_state.shape).index(3)
                                ppo_state = np.moveaxis(ppo_state, channel_axis, 1)
                                ppo_state = ppo_state.reshape(15, env.block_size, env.block_size)
                            except ValueError:
                                pass
                        
                        # 3. Case 2: 3D input (5, 40, 40) -> (15, 40, 40)
                        elif ppo_state.ndim == 3 and ppo_state.shape[0] == 5:
                            # This fixes the 8000 -> 24000 element mismatch
                            ppo_state = np.repeat(ppo_state, 3, axis=0)
                    
                    action, _states = agent.predict(ppo_state, deterministic=True)
                    action_idx = int(action)
                elif name == "reinforce":
                    action, _, _ = agent.policy.act(state, agent.device)  # handles tensor conversion internally
                    action_idx = action
            
            state, _, terminated, truncated, _ = env.step(action)
            
            terminated_by_stay = terminated and (action_idx == 0)
            done = terminated or truncated

        if terminated_by_stay:
            if env.current_patch_overlap_with_lesion() > 0:
                outcomes[name] = "hard_win"
                if name == 'dqn':
                    dqn_stats['hard_win'] += 1
                elif name == 'ppo':
                    ppo_stats['hard_win'] += 1
                elif name == 'reinforce':
                    reinforce_stats['hard_win'] += 1
            elif env.current_patch_overlap_with_lesion() == 0:
                outcomes[name] = "hard_loss"
                if name == 'dqn':
                    dqn_stats['hard_loss'] += 1
                elif name == 'ppo':
                    ppo_stats['hard_loss'] += 1
                elif name == 'reinforce':
                    reinforce_stats['hard_loss'] += 1
        else:
            if env.current_patch_overlap_with_lesion() > 0:
                outcomes[name] = "timeout_win"
                if name == 'dqn':
                    dqn_stats['timeout_win'] += 1
                elif name == 'ppo':
                    ppo_stats['timeout_win'] += 1
                elif name == 'reinforce':
                    reinforce_stats['timeout_win'] += 1
            else:
                outcomes[name] = "timeout_loss"
                if name == 'dqn':
                    dqn_stats['timeout_loss'] += 1
                elif name == 'ppo':
                    ppo_stats['timeout_loss'] += 1
                elif name == 'reinforce':
                    reinforce_stats['timeout_loss'] += 1
    return trajectories, dqn_stats, ppo_stats, reinforce_stats, outcomes

def show_the_three(trajectories, env_config, img_path, mask_path, saving_name, outcomes):    
    frames = []
    max_steps = max(len(traj) for traj in trajectories.values())
    
    temp_env = GlioblastomaPositionalEncoding(img_path, mask_path, **env_config)
    for step in range(max_steps):
        miniframes = []
        for name in trajectories.keys():
            traj = trajectories[name]
            pos = traj[step] if step < len(traj) else traj[-1]
            
            temp_env.agent_pos = pos
            temp_env.current_step = min(step, len(traj)-1)
            
            frame = temp_env.render(show=False)
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            try: 
                font = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()
            text = f"{name.upper()} {outcomes[name]} Step {min(step, len(traj)-1)}"
            draw.text((5, 25), text, fill=(0, 255, 255), font=font)
            miniframes.append(np.array(pil_img))
        # Combine miniframes side by side
        combined_frame = np.concatenate(miniframes, axis=1)
        frames.append(Image.fromarray(combined_frame))   
        
    if frames:
        # check if directory exists, if not create it
        dataset = CURRENT_CONFIG['dataset'].upper()
        os.makedirs(f"BENCHMARKING/THREE_TEST/{dataset}", exist_ok=True)
        frames[0].save(
            f"BENCHMARKING/THREE_TEST/{dataset}/{saving_name}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
                 
device = "cpu"

CURRENT_CONFIG = {
    'dataset': "glio",
    'mode': "test",
    'grid_size': 6,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        80.0,   # Correct Stay (Goal)
        -40.0,  # Wrong Stay (False Positive penalty)
        3.0,    # Move into tumor (The "Warm" hint)
        -0.5,   # Exit tumor (The "Cold" hint)
        0.05,    # Move within tumor (Encourage staying on target)
        -0.1    # Step penalty (Urgency)
    ],
    'max_steps': 75
}

# CURRENT_CONFIG = {
#     'dataset': "polyp",
#     'mode': "test",
#     'grid_size': 6,
#     'action_space': spaces.Discrete(5), 
#     'rewards': [
#         50.0,   # Correct Stay (Goal)
#         -50.0,  # Wrong Stay (False Positive penalty)
#         3.0,    # Move into tumor (The "Warm" hint)
#         -0.5,   # Exit tumor (The "Cold" hint)
#         0.05,    # Move within tumor (Encourage staying on target)
#         -0.1    # Step penalty (Urgency)
#     ],
#     'max_steps': 75
# }
    
LR = 5e-5
test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="test")
if CURRENT_CONFIG['dataset'] == 'glio':
    DQN_MODEL_NAME = "GLIO_grid_shaping_004" # 47.00% -- 80.0, -40, 3.0, -1.5, 1.0, -0.1
    PPO_MODEL_NAME = "GLIO_ppo_010" # 37.00% -- 60.0, -40, 2.5, -2.5, 0.5, -0.2
    REINFORCE_RUN_NAME = "GLIO_reinforce_011" # 35.00% -- 70.0, -70, 2.5, -2.5, 0.5, -0.1
elif CURRENT_CONFIG['dataset'] == 'polyp':
    DQN_MODEL_NAME = "POLYP_reward_shaping_008" # 37.97% -- 25.0, -12.0, 0.5, -0.5, 0.07, -0.15
    PPO_MODEL_NAME = "POLYP_ppo_009" # 44.30% -- 60.0, -40, 2.5, -2.5, 0.5, -0.2
    REINFORCE_RUN_NAME = "POLYP_reinforce_004" # 43% -- 70.0, -120, 2.5, -2.5, 0.5, -0.1
    

dqn_env = GlioblastomaPositionalEncoding(*test_pairs[0], **CURRENT_CONFIG)
dqn_model = DQNPositionalEncoding(dqn_env, learning_rate=LR, device=device, dataset=CURRENT_CONFIG['dataset'])
dqn_model.load_state_dict(
    torch.load(f"models_DQN/{DQN_MODEL_NAME}_BEST_VAL.dat", map_location=device)
)
dqn_model.eval()  # important
dqn_agent = DQNAgent(
    env_config=CURRENT_CONFIG,
    dnnetwork=dqn_model,
    buffer_class=ReplayBuffer,
    train_pairs=test_pairs,
    validation_pairs=None,
    env_class=GlioblastomaPositionalEncoding,
    epsilon=0.0                           
)

ppo_env = DatasetWrapper(
    image_paths=[p[0] for p in test_pairs],
    mask_paths=[p[1] for p in test_pairs],
    **CURRENT_CONFIG
)
ppo_model_path = f"/Users/martina/code/4year/new/models_PPO/{PPO_MODEL_NAME}/best/best_model.zip"
PPO_MODEL_NAME = PPO_MODEL_NAME + "_BEST"
ppo_loaded_model = PPO.load(ppo_model_path)
ppo_loaded_model.env_class = DatasetWrapper

reinforce_agent = REINFORCEAgent(
    env_class=GlioblastomaPositionalEncoding,
    train_pairs=test_pairs,
    env_config=CURRENT_CONFIG,
    gamma=0.99,
    lr=1e-4,
    save_path=f"models_reinforce/{REINFORCE_RUN_NAME}_best.pt"
)
reinforce_agent.policy.load_state_dict(torch.load(f"models_reinforce/{REINFORCE_RUN_NAME}_best.pt"))

# to count how many wins/losses/timeouts
total_dqn_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
total_ppo_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
total_reinforce_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}

percentage_dqn_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
percentage_ppo_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}
percentage_reinforce_stats = {'hard_win': 0, 'hard_loss': 0, 'timeout_win': 0, 'timeout_loss': 0}

for pair in test_pairs:
    print(f"Processing image: {pair[0].split('/')[-1]}")
    img_path, mask_path = pair
    # print(img_path)
    trajectories, dqn_stats, ppo_stats, reinforce_stats, outcomes = visualize_three_agents(
        img_path, mask_path,
        dqn_agent, ppo_loaded_model, reinforce_agent,
        CURRENT_CONFIG
    )
    # Now you can use 'trajectories' to visualize or analyze the paths taken by each agent
    saving_name = f"{img_path.split('/')[-1].split('.')[0]}_three_agents"
    # print(f"Saving {saving_name}.gif")
    show_the_three(trajectories, CURRENT_CONFIG, img_path, mask_path, saving_name, outcomes)
    
    for key in dqn_stats:
        total_dqn_stats[key] += dqn_stats[key]
    for key in ppo_stats:
        total_ppo_stats[key] += ppo_stats[key]
    for key in reinforce_stats:
        total_reinforce_stats[key] += reinforce_stats[key]

print("DQN Stats:", total_dqn_stats)
print("PPO Stats:", total_ppo_stats)
print("REINFORCE Stats:", total_reinforce_stats)

for key in total_dqn_stats:
    #keep only two decimal places
    percentage_dqn_stats[key] = str(round(total_dqn_stats[key] / len(test_pairs) * 100, 2)) + "%"
    percentage_ppo_stats[key] = str(round(total_ppo_stats[key] / len(test_pairs) * 100, 2)) + "%"
    percentage_reinforce_stats[key] = str(round(total_reinforce_stats[key] / len(test_pairs) * 100, 2)) + "%"
print("========================= Percentages ========================")
print(f"DQN Percentages: {percentage_dqn_stats}")
print(f"PPO Percentages: {percentage_ppo_stats}")
print(f"REINFORCE Percentages: {percentage_reinforce_stats}")