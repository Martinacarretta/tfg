import numpy as np
import matplotlib.pyplot as plt

import torch
from copy import deepcopy
import wandb
import random

from tqdm.auto import tqdm

SEED = 42

class DQNAgent:
    
    def __init__(self, env_config, dnnetwork, buffer_class, train_pairs, env_class, 
                 epsilon=0.1, eps_decay=0.99, eps_decay_type="subtraction", epsilon_min=0.01, batch_size=32, gamma=0.99, 
                 memory_size=1500, buffer_initial=150, save_name="Glioblastoma"):
        self.env_config = env_config
        self.env_class = env_class
        
        self.dnnetwork = dnnetwork # main network
        self.target_network = deepcopy(dnnetwork) # prevents the target Q-values from changing with every single update
        self.target_network.optimizer = None # paper said target net is only  weights, no optimizer

        self.epsilon = epsilon # initial epsilon for e-greedy
        self.eps_decay = eps_decay # decay of epsilon after each episode to balance exploration and exploitation
        self.eps_decay_type = eps_decay_type

        self.epsilon_min = 0 if self.epsilon == 0 else epsilon_min
            
        self.batch_size = batch_size # size of the mini-batch for training
        self.gamma = gamma
        
        self.buffer_initial = buffer_initial # number of random experiences to fill the buffer before training
        
        self.save_name = save_name
        
        # block of the last X episodes to calculate the average reward 
        self.nblock = 100 
                
        # Buffer
        self.buffer = buffer_class(capacity=memory_size)

        self.initialize()
        
    
    def initialize(self): # reset variables at the beginning of training
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = None  # Initialize as None since we don't have env yet
        
    ## Take new action
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # random action in burn-in and in the exploration phase (epsilon)
            action = self.env.action_space.sample() 
        else:
            # Action based on the Q-value (max Q-value)
            action = self.dnnetwork.get_action(self.state0, eps)
            self.step_count += 1
            
        # Execute action and get reward and new state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        
        # save experience in the buffer
        self.buffer.append(self.state0, action, reward, done, new_state)
        self.state0 = new_state.copy()
        
        if done:
            self.state0 = self.env.reset()[0]
        return done, reward # THE REWARD RETURN IS FOR DEBUGGING 

            
    ## Training
    def train(self, train_pairs, gamma=0.99, max_episodes=50000, 
              dnn_update_frequency=4,
              dnn_sync_frequency=200):
        
        self.gamma = gamma

        # Fill the buffer with N random experiences
        print("Filling replay buffer...")
        
        first = True # just to detect the input channel number
        
        for img_path, mask_path in train_pairs:
            self.env = self.env_class(img_path, mask_path, **self.env_config)
            self.state0, _ = self.env.reset(seed=SEED)
            if first:
                first = False
                # UPDATED: Detect input channels from first environment so i can use with both envs and dqns
                if self.state0.ndim == 2:
                    self.input_channels = 1 # in case it were (60, 60)
                else:
                    self.input_channels = self.state0.shape[0] # in case it were (3, 60, 60)
            # Run short episode on this image
            for _ in range(self.buffer_initial):
                self.take_step(self.epsilon, mode='explore')

        print(f"Buffer filled with {len(self.buffer.buffer)} experiences")

            
        # Store metrics locally to plot
        self.episode_rewards = []
        self.mean_rewards = []
        self.epsilon_values = []
        self.loss_values = []
 
        episode = 0
        training = True
        
        pbar = tqdm(total=max_episodes, desc="Initializing", 
                unit="ep", unit_scale=True, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        
        print("Training...")
        while training and episode < max_episodes:
            img_path, mask_path = random.choice(train_pairs)
            self.env = self.env_class(img_path, mask_path, **self.env_config)
            self.state0, _ = self.env.reset(seed=SEED)
            self.total_reward = 0
            
            # DEBUGGING
            pos_rewards = 0
            neg_rewards = 0
            
            for step in range(self.env.max_steps):
                gamedone, reward = self.take_step(self.epsilon, mode='train') # THE REWARD RETURN IS FOR DEBUGGING

                # DEBUGGING
                if reward > 0: pos_rewards += 1
                else: neg_rewards += 1
                # END OF DEBUGGING
                
                # Upgrade main network
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                    
                # Synchronize the main network and the target network
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)
                    
                if gamedone:
                    episode += 1
                    pbar.update(1)
          
                    # Save the rewards
                    self.training_rewards.append(self.total_reward)
                    # Calculate the average reward for the last X episodes
                    if len(self.training_rewards) >= self.nblock:
                        mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    else:
                        mean_rewards = np.mean(self.training_rewards)  # Use all rewards if less than nblock
                    
                    self.mean_training_rewards.append(mean_rewards)

                    print("Episode {:d} | Episode reward {:.2f} | Mean Rewards {:.2f} | Epsilon {:.4f} | Loss {:.4f}".format(
                        episode, self.total_reward, mean_rewards, self.epsilon, np.mean(self.update_loss)))
                    print(f"      Positive rewards: {pos_rewards}, Negative rewards: {neg_rewards}") # DEBUGGING
                    
                    wandb.log({
                        'episode': episode,
                        'mean_rewards': mean_rewards,
                        'episode reward': self.total_reward,
                        'epsilon': self.epsilon,
                        'loss': np.mean(self.update_loss)
                    }, step=episode)
                    
                    # Append metrics to lists for plotting
                    self.episode_rewards.append(self.total_reward)
                    self.mean_rewards.append(mean_rewards)
                    self.epsilon_values.append(self.epsilon)
                    self.loss_values.append(np.mean(self.update_loss))
                    
                    self.update_loss = []

                    # Check if there are still episodes left
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    if self.eps_decay_type == "exponential":
                        # Update epsilon according to exponential decay
                        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
                    else:
                        self.epsilon = max(self.epsilon - self.eps_decay, self.epsilon_min) 
                        
                    torch.save(self.dnnetwork.state_dict(), self.save_name + ".dat")
        
        pbar.close()
                    
        # PLOTTING
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
        axes = axes.ravel()  # Flatten the axes array for easier indexing

        # Plot episode rewards
        axes[0].plot(self.episode_rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Episode Rewards Over Time')

        # Plot mean rewards
        axes[1].plot(self.mean_rewards)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Mean Reward')
        axes[1].set_title('Mean Rewards Over Time')

        # Plot epsilon values
        axes[2].plot(self.epsilon_values)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Epsilon Values')
        axes[2].set_title('Epsilon Values Over Time')

        # Plot loss values
        axes[3].plot(self.loss_values)
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Loss Over Time')

        # Adjust layout
        # fig.suptitle('Training Performance', fontsize=16)  # Add a main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust spacing between subplots and title
        plt.show()


    # Loss calculation           
    def calculate_loss(self, batch):
        # Separate the variables of the experience and convert them to tensors
        states, actions, rewards, dones, next_states = batch
        device = self.dnnetwork.device

        # Add channel dimension # FALTAA
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        # If grayscale (H,W) was stored → convert to (1,H,W)
        if self.input_channels == 1 and states.ndim == 3:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

        rewards_vals = torch.FloatTensor(rewards).to(device=device) 
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=device)
        dones_t = torch.BoolTensor(dones).to(device=device)
        
        # Obtain the Q values of the main network
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)
        
        # Obtain the target Q values.
        # The detach() parameter prevents these values from updating the target network
        qvals_next_all = self.target_network.get_qvals(next_states)  # Shape: [batch_size, n_actions]
        qvals_next = torch.max(qvals_next_all, dim=1)[0].detach()    # Shape: [batch_size]

        # 0 in terminal states
        qvals_next[dones_t] = 0.0 
        
        # print("qvals_next.shape", qvals_next.shape, "dones_t.shape", dones_t.shape) # debugging
        
        # Calculate the Bellman equation
        expected_qvals = (self.gamma * qvals_next) + rewards_vals
        
        # Calculate the loss
        # loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        # Use Huber loss instead of MSELoss
        loss = torch.nn.SmoothL1Loss()(qvals, expected_qvals.reshape(-1,1))
        return loss
    

    def update(self, num_buffers=4):
        # Check if buffer has enough experiences
        if len(self.buffer.buffer) < self.batch_size:
            return
             
        # Remove any gradient
        self.dnnetwork.optimizer.zero_grad()  
        
        batch = self.buffer.sample_batch(self.batch_size)

        # Calculate the loss
        loss = self.calculate_loss(batch) 
        # Difference to get the gradients
        loss.backward() 
        
        # add gradient clipping
        # Clipping uses L2 of gradients - “Don’t let gradients go crazy. Keep training stable.”
        # Regularization uses L2 of weights - “Don’t let weights get too big. Keep the model simple.”
        torch.nn.utils.clip_grad_norm_(self.dnnetwork.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.dnnetwork.parameters(), max_norm=0.1)
        # Apply the gradients to the neural network
        self.dnnetwork.optimizer.step() 
        
        # Save loss values
        if self.dnnetwork.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

