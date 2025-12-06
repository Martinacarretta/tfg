import numpy as np
import torch
import torch.nn as nn

SEED = 42

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
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
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
    
