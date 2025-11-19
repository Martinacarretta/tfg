import numpy as np
import torch
import torch.nn as nn

SEED = 42

"""
Original DQN:
    Input: grayscale image (1, 60, 60)
    Architecture: 4 conv layers (32 filters each) + 4 fc layers (512-256-128-output)
    Activation: ELU
    
DQN2:
    Input: image + 2 position channels (3, 60, 60)
    Architecture: same as DQN (but obv adapated for the input which has 3 channels)
    Activation: ELU
    
"""

class DQNPaper(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQNPaper, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0] # 60
        self.n_outputs = env.action_space.n # 3
        self.actions = np.arange(env.action_space.n) # np.array([0, 1, 2])
        self.learning_rate = learning_rate
        
        input_channels = 1
        height, width = env.observation_space.shape  # 60, 60   
        
        ### Construction of the neural network
        ## features first and then fully connected layers
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        
        # flatten 
        with  torch.no_grad(): # FALTA MIRAR Q ES AIXO
            dummy_input = torch.zeros(1, input_channels, height, width) # batch size 1
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            
        # nn.Linear (in features, out features)
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.n_outputs)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        ### Work with CUDA is allowed
        if self.device == 'cuda':
            self.to(self.device).cuda()
            
    def forward(self, x):
        # x shape: (batch_size, 1, 60, 60)
        features = self.features(x)
        features_flat = features.view(x.size(0), -1)
        q_values = self.fc(features_flat)
        return q_values
    
    # e-greedy method
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            # random action -- Exploration
            action = np.random.choice(self.actions)  
        else:
            # Q-value based action -- Exploitation
            qvals = self.get_qvals(state)  
            if qvals.dim() == 2 and qvals.size(0) == 1:
                action = torch.argmax(qvals, dim=-1).item()
            else:
                action = torch.argmax(qvals, dim=-1)[0].item()

        return int(action)
    
    # forward pass through conv and fc layers
    def get_qvals(self, state):
        if isinstance(state, np.ndarray):
            if state.ndim == 2:
                state = state[np.newaxis, np.newaxis, :, :]  # (1,1,60,60)
            elif state.ndim == 3:
                if state.shape[-1] == 60:  # (batch, 60, 60)
                    state = state[:, np.newaxis, :, :]  # (batch,1,60,60)
        state_t = torch.FloatTensor(state).to(self.device)
        return self.forward(state_t)
    
class DQNPositionalEncoding(torch.nn.Module): #DQN2
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQNPositionalEncoding, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # UPDATED: Now has 3 input channels (image + 2 position channels)
        input_channels = 3  # Changed from 1 to 3
        height, width = env.observation_space.shape[1], env.observation_space.shape[2]  # (60, 60)
        
        ### Construction of the neural network
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.n_outputs)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.device == 'cuda':
            self.to(self.device).cuda()
            
    def forward(self, x):
        # x shape: (batch_size, 3, 60, 60)  # Updated comment
        features = self.features(x)
        features_flat = features.view(x.size(0), -1)
        q_values = self.fc(features_flat)
        return q_values
    
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            if qvals.dim() == 2 and qvals.size(0) == 1:
                action = torch.argmax(qvals, dim=-1).item()
            else:
                action = torch.argmax(qvals, dim=-1)[0].item()

        return int(action)
    
    def get_qvals(self, state):
        # UPDATED: Handle (3, 60, 60) observations
        if isinstance(state, np.ndarray):
            if state.ndim == 3:  # Single observation (3, 60, 60)
                state = np.expand_dims(state, 0)  # (1, 3, 60, 60)
            elif state.ndim == 4:  # Already batched (batch, 3, 60, 60)
                pass
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
                
        state_t = torch.FloatTensor(state).to(self.device)
        qvals = self.forward(state_t)
        return qvals
    

class DQNMriLite(nn.Module):
    """
    Stable CNN for MRI RL navigation.
    Works with 60x60, 30x30, and multi-channel input.
    No BatchNorm, No Dropout, No Residuals (DQN-stable).
    """

    def __init__(self, env, learning_rate=1e-4, device="cpu"):
        super().__init__()
        self.device = device
        self.n_actions = env.action_space.n
        self.actions = np.arange(self.n_actions)
        self.learning_rate = learning_rate

        obs_shape = env.observation_space.shape

        # Vision-only: (H, W)
        if len(obs_shape) == 2:
            channels = 1
            height, width = obs_shape

        # Positional encoding: (C, H, W)
        elif len(obs_shape) == 3:
            channels, height, width = obs_shape

        else:
            raise ValueError(f"Unexpected obs shape: {obs_shape}")

        # ===========
        # CNN BLOCKS
        # ===========

        # Block 1: reduce 60→30 or 30→15
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # Block 2: feature extraction without reducing H,W
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Block 3: compress 30→15 or 15→8 (only 1 more reduction)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            out = self.conv3(self.conv2(self.conv1(dummy)))
            n_flat = out.view(1, -1).size(1)

        # =================
        # FULLY CONNECTED
        # =================
        # self.fc = nn.Sequential(
        #     nn.Linear(n_flat, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_actions)
        # )
        
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ELU(),

            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, self.n_actions),
            nn.Tanh()
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if device == "cuda":
            self.cuda()

    # ====================
    # FORWARD + GET Q-VALS
    # ====================
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_qvals(self, state):
        # Convert numpy → tensor
        if isinstance(state, np.ndarray):
            if state.ndim == 2:
                state = state[np.newaxis, np.newaxis, :, :]  # (1,1,H,W)
            elif state.ndim == 3:
                state = state[np.newaxis, :, :, :]           # (1,C,H,W)

        state_t = torch.FloatTensor(state).to(self.device)
        return self.forward(state_t)   # IMPORTANT: NO no_grad

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            return int(np.random.choice(self.actions))

        with torch.no_grad():
            qvals = self.get_qvals(state)

        return int(torch.argmax(qvals, dim=-1).item())

