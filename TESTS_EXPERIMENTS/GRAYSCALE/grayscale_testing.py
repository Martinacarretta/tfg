import numpy as np
import random
import torch
import gymnasium as gym
from gymnasium import spaces

from grayscale_general import prepare, testing
from env import GlioblastomaPositionalEncoding
from training_dqnpos import DQNPositionalEncoding
from training_agents import DQNAgent
from training_buffers import ReplayBuffer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GrayscalePolypWrapper(gym.Wrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
    
    def render(self, show=False):
        frame = self.env.render(show=False)

        # 1. Convert whole frame to grayscale
        gray = (
            0.299 * frame[..., 0]
            + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]
        ).clip(0, 255).astype(np.uint8)

        gray_rgb = np.stack([gray] * 3, axis=-1)

        # 2. ERASE env-drawn agent box + text region
        r0 = self.env.agent_pos[0] * self.env.block_size
        c0 = self.env.agent_pos[1] * self.env.block_size
        bs = self.env.block_size

        t = 2
        yellow = np.array([255, 255, 0], dtype=np.uint8)

        # Top
        gray_rgb[r0:r0+t, c0:c0+bs] = yellow
        # Bottom
        gray_rgb[r0+bs-t:r0+bs, c0:c0+bs] = yellow
        # Left
        gray_rgb[r0:r0+bs, c0:c0+t] = yellow
        # Right
        gray_rgb[r0:r0+bs, c0+bs-t:c0+bs] = yellow

        return gray_rgb




    
CURRENT_CONFIG = {
    'dataset': "polyp",
    'mode': "test",
    'grid_size': 4,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        20.0,   # Correct Stay (Goal)
        -6.0,  # Wrong Stay (False Positive penalty)
        0.5,    # Move into tumor (The "Warm" hint)
        -0.5,   # Exit tumor (The "Cold" hint)
        0.1,    # Move within tumor (Encourage staying on target)
        -0.1    # Step penalty (Urgency)
    ],
    'max_steps': 50
}
    
start = True  # Whether to start on zero or random position
best = True
MODEL_NAME = "grayscale"
LR = 5e-5

test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="test")
device = "cpu"
raw_temp_env = GlioblastomaPositionalEncoding(*test_pairs[0], **CURRENT_CONFIG)
grayscale_env = GrayscalePolypWrapper(raw_temp_env)

model = DQNPositionalEncoding(grayscale_env, learning_rate=LR, device=device, dataset=CURRENT_CONFIG['dataset'])

if best:
    MODEL_NAME = MODEL_NAME + "_BEST_VAL"
    model.load_state_dict(
        torch.load(f"models_DQN/{MODEL_NAME}.dat", map_location=device)
    )
else:
    model.load_state_dict(
        torch.load(f"models_DQN/{MODEL_NAME}.dat", map_location=device)
    )

model.eval()  # important

def wrapped_env_creator(*args, **kwargs):
    raw = GlioblastomaPositionalEncoding(*args, **kwargs)
    return GrayscalePolypWrapper(raw)

agent = DQNAgent(
    env_config=CURRENT_CONFIG,
    dnnetwork=model,
    buffer_class=ReplayBuffer,
    train_pairs=test_pairs,
    validation_pairs=None,
    env_class=wrapped_env_creator,  # ✅ WRAPPED
    epsilon=0.0
)


if start:
    results2 = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="dqn",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"BENCHMARKING/GRAYSCALE/SOZ_GIFs_Testing_{MODEL_NAME}",
        start_on_zero=True, 
        print_all=False
    )
else:
    results = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="dqn",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"BENCHMARKING/GRAYSCALE/GIFs_Testing_{MODEL_NAME}", 
        print_all=False
    )


print("Testing completed.")