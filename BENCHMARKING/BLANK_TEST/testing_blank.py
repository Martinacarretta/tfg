#### TEST IF BEST DQN MODEL ON GLIO WORKS WHEN A PATCH OF THE CENTER IS BLANK ####  

import numpy as np
import random
import torch
from gymnasium import spaces

from general import prepare, testing
from env_blank import GlioblastomaPositionalEncoding
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


CURRENT_CONFIG = {
    'dataset': "glio",
    'mode': "test",
    'grid_size': 4,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        30.0,   # Correct Stay (Goal)
        -15.0,  # Wrong Stay (False Positive penalty)
        0.5,    # Move into tumor (The "Warm" hint)
        -0.5,   # Exit tumor (The "Cold" hint)
        0.05,    # Move within tumor (Encourage staying on target)
        -0.2    # Step penalty (Urgency)
    ],
    'max_steps': 50
}
    
start = True  # Whether to start on zero or random position
best = True
MODEL_NAME = "GLIO_reward_shaping_008"
LR = 5e-5

test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="test")
device = "cpu"
env = GlioblastomaPositionalEncoding(*test_pairs[0], **CURRENT_CONFIG)
model = DQNPositionalEncoding(env, learning_rate=LR, device=device, dataset=CURRENT_CONFIG['dataset'])

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

agent = DQNAgent(
    env_config=CURRENT_CONFIG,
    dnnetwork=model,
    buffer_class=ReplayBuffer,
    train_pairs=test_pairs,
    validation_pairs=None,
    env_class=GlioblastomaPositionalEncoding,
    epsilon=0.0                           
)

results2 = testing(
    agent=agent,
    test_pairs=test_pairs,
    agent_type="dqn",
    num_episodes=len(test_pairs),
    env_config=CURRENT_CONFIG,
    save_gifs=True,
    gif_folder=f"GIF_DQN_BLANK/SOZ_GIFs_Testing_BLANK",
    start_on_zero=True, 
    print_all=False
)

print("Testing completed.")