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
    
DQN3:
    Input: image + 2 position channels (3, 60, 60)
    Architecture: SIMPLIFIED!!! 3 conv layers (16,32,32 filters)
    FC layers: 128-64-output
    Activation: ReLU
    Weight init: Xavier with gain 0.5
    Stability! 

DQN4: 
    Input: 3 channels
    Architecture: same as DQN3
    FC layers: same as DQN3
    Adam with weight decay (L2 regularization) for stability
    Tries to prevent overfitting/exploding weights
    
DQN5:
    Input: 3 channels
    Architecture: SIMPLER even! 3 conv layers (16,32,32 filters) with BatchNorm
    FC layers: 64-32-output with Dropout
    AdamW with weight decay
    Weight init: Kaiming normal for ReLU
    

"""

class DQN(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN, self).__init__()
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
        # Convert (60,60) → (1,1,60,60)
        if isinstance(state, np.ndarray):
            if state.ndim == 2:  # grayscale single image (60x60)
                state = np.expand_dims(np.expand_dims(state, 0), 0) # (1,1,60,60)
                
            elif state.ndim == 3:  # batch or stacked images (batch, 60, 60)
                if state.shape[0] != 1: #batch
                    state = np.expand_dims(state, 1)
                    
        state_t = torch.FloatTensor(state).to(self.device)
        qvals = self.forward(state_t)  # Use the forward method instead
        return qvals
    
class DQN2(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN2, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # UPDATED: Now accepting 3 input channels (image + 2 position channels)
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
    
class DQN3(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN3, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        input_channels = 3
        height, width = env.observation_space.shape[1], env.observation_space.shape[2]
        
        ### SIMPLIFIED: Fewer layers, smaller network
        self.features = nn.Sequential(
            # 3x60x60 -> 16x30x30
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16x30x30 -> 32x15x15
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 32x15x15 -> 32x8x8
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            print(f"Flattened feature size: {n_flatten}")
            
        ### SIMPLIFIED: Smaller FC layers
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128),  # Reduced from 512
            nn.ReLU(),
            nn.Linear(128, 64),  # Reduced from 256
            nn.ReLU(),
            nn.Linear(64, self.n_outputs)
        )
        
        # Initialize weights with smaller values for stability
        self.apply(self._init_weights)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.device == 'cuda':
            self.to(self.device).cuda()
    
    def _init_weights(self, module):
        """Initialize weights with smaller values for stability"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
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
        if isinstance(state, np.ndarray):
            if state.ndim == 3:  # Single obs (3, 60, 60)
                state = np.expand_dims(state, 0)  # (1, 3, 60, 60)
            elif state.ndim == 4:  # Batch (batch, 3, 60, 60)
                pass
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
                
        state_t = torch.FloatTensor(state).to(self.device)
        qvals = self.forward(state_t)
        return qvals

class DQN4(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN4, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        input_channels = 3
        height, width = env.observation_space.shape[1], env.observation_space.shape[2]
        
        ### SIMPLIFIED: Fewer layers, smaller network
        self.features = nn.Sequential(
            # 3x60x60 -> 16x30x30
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16x30x30 -> 32x15x15
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 32x15x15 -> 32x8x8
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            print(f"Flattened feature size: {n_flatten}")
            
        ### SIMPLIFIED: Smaller FC layers
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128),  # Reduced from 512
            nn.ReLU(),
            nn.Linear(128, 64),  # Reduced from 256
            nn.ReLU(),
            nn.Linear(64, self.n_outputs)
        )
        
        # Initialize weights with smaller values for stability
        self.apply(self._init_weights)
        
        # Use Adam with weight decay (L2 regularization) for stability
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # Prevent weights from exploding
        )
        
        if self.device == 'cuda':
            self.to(self.device).cuda()
    
    def _init_weights(self, module):
        """Initialize weights with smaller values for stability"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
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
        if isinstance(state, np.ndarray):
            if state.ndim == 3:  # Single obs (3, 60, 60)
                state = np.expand_dims(state, 0)  # (1, 3, 60, 60)
            elif state.ndim == 4:  # Batch (batch, 3, 60, 60)
                pass
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
                
        state_t = torch.FloatTensor(state).to(self.device)
        qvals = self.forward(state_t)
        return qvals

class DQN5(torch.nn.Module):
    """Much simpler architecture for stable training"""
    
    def __init__(self, env, learning_rate=1e-5, device='cpu'):
        super(DQN5, self).__init__()
        self.device = device
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        input_channels = 3
        height, width = env.observation_space.shape[1], env.observation_space.shape[2]
        
        # Very simple feature extraction
        self.features = nn.Sequential(
            # 3x60x60 -> 16x30x30
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # 16x30x30 -> 32x15x15
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x15x15 -> 32x8x8
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)
            print(f"Flattened feature size: {n_flatten}")
            
        # Simple FC layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs)
        )
        
        # Conservative weight initialization
        self.apply(self._init_weights)
        
        # Adam with very low learning rate and weight decay
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        if self.device == 'cuda':
            self.to(self.device)
    
    def _init_weights(self, module):
        """Conservative weight initialization"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(x.size(0), -1)
        q_values = self.fc(features_flat)
        return q_values
    
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            with torch.no_grad():
                qvals = self.get_qvals(state)
                action = torch.argmax(qvals, dim=-1).item()
        return int(action)
    
    def get_qvals(self, state, for_training=True):
        """Get Q-values with optional gradient tracking"""
        if isinstance(state, np.ndarray):
            if state.ndim == 3:  # Single obs (3, 60, 60)
                state = np.expand_dims(state, 0)  # (1, 3, 60, 60)
            elif state.ndim == 4:  # Batch (batch, 3, 60, 60)
                pass
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
                
        state_t = torch.FloatTensor(state).to(self.device)
        
        if for_training:
            # During training, keep the computation graph
            qvals = self.forward(state_t)
        else:
            # During action selection or target computation, use no_grad
            with torch.no_grad():
                qvals = self.forward(state_t)
        return qvals