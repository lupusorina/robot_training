import ast

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import gymnasium as gym



class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)



#_________________________________________________________ Single Gaussian MLE _________________________________________________________________

class ContinuousActionNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousActionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, action_dim)
        self.log_sigma_head = nn.Linear(32, action_dim)  

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

def gaussian_nll_loss(mu, log_std, target):
    std = torch.exp(log_std)
    variance = std ** 2
    log_variance = 2 * log_std
    
    nll = 0.5 * (
        log_variance + 
        ((target - mu) ** 2) / variance + 
        torch.log(2 * torch.tensor(np.pi))
    )
    return nll.mean()

def train_mle_with_val(model: ContinuousActionNN,
                       train_loader: DataLoader,
                       lr: float,
                       num_epochs: int,
                       device: str = "cpu",
                       val_split: float = 0.2,
                       BestLoss: float = 1e6):
    
    data = train_loader.dataset
    total_size = len(data)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_subset, val_subset = random_split(data, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True, 
                              num_workers=train_loader.num_workers)
    val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False,
                            num_workers=train_loader.num_workers)
    
    train_losses = []
    val_losses = []
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        i = 0
        for input_batch, actions_batch in train_loader:
            # print(f'\ncurrent batch: {i+1}')
            i += 1
            input_batch = input_batch.to(device)
            actions_batch = actions_batch.to(device)
            # print(f'input_batch: {input_batch},\nactions_batch: {actions_batch}')
            optimizer.zero_grad()
            
            # Compute negative log likelihood
            mu,log_std = model(input_batch)
            loss = gaussian_nll_loss(mu, log_std, actions_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            
            total_loss += loss.item() * actions_batch.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_batch, actions_batch in val_loader:
                input_batch = input_batch.to(device)
                actions_batch = actions_batch.to(device)
    
                mu, log_std = model(input_batch)
                loss = gaussian_nll_loss(mu, log_std, actions_batch)
                total_val_loss += loss.item() * actions_batch.size(0)
        
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        if val_loss < BestLoss:
            BestLoss = val_loss
            print(f"New best validation loss: {BestLoss:.4f}")
            torch.save(model.state_dict(), f"{best_model_folder}/P(a|s).pth")

        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f} || Val Loss: {val_loss:.4f}")

    return train_losses, val_losses



STATES_PATH = '~/Desktop/MTData/20250404-103317/state_for_ddpg.csv'
CNTRL_PATH = '~/Desktop/MTData/20250404-103317/ctrl_for_ddpg.csv'
INPUT_TYPE = 'state'#'state_action' or 'state' or 'prev_state_action'
SEED = 42

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_folder = os.path.join(current_dir, 'MLE_model')
    os.makedirs(best_model_folder, exist_ok=True)

    robot_states = pd.read_csv(STATES_PATH, index_col=0)
    robot_actions = pd.read_csv(CNTRL_PATH, index_col=0)
    
    robot_state_dim = robot_states.shape[1] #47
    nb_state_samples = robot_states.shape[0] #14
    
    robot_action_dim = robot_actions.shape[1]
    nb_action_samples = robot_actions.shape[0]
    
    assert nb_state_samples == nb_action_samples, "Number of samples in states and actions must be equal"
    state_tensor = torch.tensor(robot_states.values, dtype=torch.float32)
    action_tensor = torch.tensor(robot_actions.values, dtype=torch.float32)
    print(f"State tensor shape: {state_tensor.shape}")
    print(f"Action tensor shape: {action_tensor.shape}")

    
    # last 20% used for testing, no shuffle to check rollout.
    num_samples = state_tensor.shape[0]  
    train_size = int(0.8 * num_samples)  
    test_size  = num_samples - train_size 

    train_states = state_tensor[:train_size] 
    train_actions = action_tensor[:train_size] 

    test_states  = state_tensor[train_size:]  
    test_actions = action_tensor[train_size:] 
    

    train_dataset = TensorDataset(train_states, train_actions)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('data ready')
    


    model = ContinuousActionNN(robot_state_dim, robot_action_dim)
    train_losses, val_episodes = train_mle_with_val(model=model, train_loader=train_loader, lr=1e-3,
                                      num_epochs= 150, device= "cpu", val_split = 0.2,
                                      BestLoss = 1e6)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_episodes, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL Loss')
    # plt.savefig(f'Data/Plots/{model_name}_Loss.svg')
    plt.show()
    quit()
    model.eval()

    
    val_states = data[data['episode'].isin(val_episodes)]['obs'].to_numpy()
    val_states = [np.array(s, dtype=np.float32) for s in val_states]
    val_states = np.stack(val_states)
    val_states_tensor = torch.tensor(val_states, dtype=torch.float32)
    val_actions = data[data['episode'].isin(val_episodes)]['action'].to_numpy()
    val_actions = [np.array(a, dtype=np.float32) for a in val_actions]
    val_actions = np.stack(val_actions)

    # val_prev_th = data[data['episode'].isin(val_episodes)]['angle_state'].to_numpy()
    # val_prev_th_d = data[data['episode'].isin(val_episodes)]['angle_vel'].to_numpy()
    # val_prev_states = np.array([val_prev_th, val_prev_th_d]).T
    # val_prev_states_tensor = torch.tensor(val_prev_states, dtype=torch.float32)
    # val_prev_actions = data[data['episode'].isin(val_episodes)]['prev_action'].to_numpy()
    # val_prev_actions_tensor = torch.tensor(val_prev_actions.reshape(-1,1), dtype=torch.float32)

    if INPUT_TYPE == 'state':
        model_input = val_states_tensor
    # elif INPUT_TYPE == 'state_action':
    #     model_input = torch.cat((val_states_tensor, val_prev_actions_tensor), dim=1)
    # elif INPUT_TYPE == 'prev_state_action':
    #     model_input = torch.cat((val_states_tensor, val_prev_actions_tensor, val_prev_states_tensor), dim=1)
            

    with torch.no_grad():
        mu, log_std = model(model_input)
    print(f'avg pred {mu}')
    actions_pred = mu.numpy().flatten()
    print(f'flattened {actions_pred}')
    print(f'labels {val_actions.flatten()}')
    action_pred = np.clip(actions_pred, -1, 1)
    std = torch.exp(log_std).numpy().flatten()
    diff = val_actions.flatten() - actions_pred
    avg_diff = np.mean(diff)
    print(f"Average difference between actions and predicted actions: {avg_diff}")

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('Theta')
    # ax.set_ylabel('Theta_dot')
    # ax.set_zlabel('Action')
    # plt.savefig(f'Data/Plots/{model_name}_fit.svg')
    # plt.show()
