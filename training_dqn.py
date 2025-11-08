import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from collections import deque
from copy import deepcopy
import wandb
import random
import datetime

import torch.nn as nn
import pandas as pd

from tqdm.auto import tqdm

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