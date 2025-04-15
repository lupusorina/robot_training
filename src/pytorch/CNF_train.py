import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

STATES_PATH = '20250404-103317/state_for_ddpg.csv'
CNTRL_PATH = '20250404-103317/ctrl_for_ddpg.csv'
INPUT_TYPE = 'state'#'state_action' or 'state' or 'prev_state_action'
SEED = 42
device = 'cpu'
BestLoss = 1e6

np.random.seed(42)

class ConditionalBase(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 2 * latent_dim) 
        )
    
    def forward(self, condition):
        params = self.net(condition)  
        mean = params[:, :self.latent_dim]
        log_std = params[:, self.latent_dim:]
        return mean, log_std
    

class ConditionalAffineLayer(nn.Module):
    def __init__(self, condition_dim, action_dim):  # <-- CHANGED
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 2 * action_dim)  # <-- CHANGED: was 2, now 2*action_dim
        )
    
    def forward(self, a, condition):
        """
        :param a:      shape (batch_size, action_dim)
        :param condition: shape (batch_size, condition_dim)
        :return: z, log_det
        """
        params = self.net(condition)  # shape (batch_size, 2*action_dim)
        s = params[:, :self.action_dim]     # log-scale parameters, shape (batch_size, action_dim)
        t = params[:, self.action_dim:]     # translation parameters, shape (batch_size, action_dim)
        scale = torch.exp(s)
        
        # Forward transform: a -> z = (a - t) / scale
        z = (a - t) / scale
        
        # log_det for each dimension = -log(scale), sum across action_dim
        log_det = -torch.sum(s, dim=1)  # s is log(scale)
        return z, log_det
    
    def inverse(self, z, condition):
        """
        :param z: shape (batch_size, action_dim)
        :param condition: shape (batch_size, condition_dim)
        :return: a, log_det
        """
        params = self.net(condition)
        s = params[:, :self.action_dim]
        t = params[:, self.action_dim:]
        scale = torch.exp(s)

        # Inverse transform: z -> a = z * scale + t
        a = z * scale + t
        
        # log_det for each dimension = +log(scale), sum across action_dim
        log_det = torch.sum(s, dim=1)
        return a, log_det
    

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, condition_dim, n_flows, latent_dim=1):
        """
        """
        super().__init__()
        self.n_flows = n_flows
        # Each flow layer must know the action dimension
        self.layers = nn.ModuleList([
            ConditionalAffineLayer(condition_dim, latent_dim)  # <-- CHANGED
            for _ in range(n_flows)
        ])
        self.conditional_base = ConditionalBase(condition_dim, latent_dim)
    
    def forward(self, a, condition):
        """
        Forward pass: map action a -> latent z
        Returns z, log_det
        """
        log_det_total = 0.0
        z = a
        for layer in self.layers:
            z, log_det = layer(z, condition)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z, condition):
        """
        Inverse pass: map latent z -> action a
        Returns a, log_det
        """
        log_det_total = 0.0
        a = z
        # Must invert in reverse order
        for layer in reversed(self.layers):
            a, log_det = layer.inverse(a, condition)
            log_det_total += log_det
        return a, log_det_total
    
    def log_prob(self, a, condition):
        """
        Negative log-likelihood in latent space + log_det of transform
        """
        # Transform actions to z in latent space
        z, log_det = self.forward(a, condition)
        
        # Evaluate log_prob of z under base_dist
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        # print(f"base_mean: {base_mean}, base_log_std: {base_log_std}")
        base_dist = torch.distributions.Normal(base_mean, base_std)  
        
        
        log_base = base_dist.log_prob(z).sum(dim=1)  # <-- CHANGED (sum over action_dim)
        return log_base + log_det
    
    def sample(self, num_samples, condition):
        """
        Sample from the flow given a condition, by first sampling from the
        base distribution then applying the inverse flow transform.
        """
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        base_dist = torch.distributions.Normal(base_mean, base_std)
        
        z = base_dist.rsample()  # (batch_size, action_dim)
        a, _ = self.inverse(z, condition)
        return a



    

def train_cnf_with_val(model: ConditionalNormalizingFlow,
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
            log_prob = model.log_prob(actions_batch, input_batch)
            loss = -log_prob.mean()
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
    
                log_prob = model.log_prob(actions_batch, input_batch)
                loss = -log_prob.mean()
                total_val_loss += loss.item() * actions_batch.size(0)
        
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        if val_loss < BestLoss:
            BestLoss = val_loss
            # print(f"New best validation loss: {BestLoss:.4f}")
            torch.save(model.state_dict(), f"{best_model_folder}/CNF(a|s).pth")

        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f} || Val Loss: {val_loss:.4f}")

    return train_losses, val_losses



if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_folder = os.path.join(current_dir, 'CNF_model')
    os.makedirs(best_model_folder, exist_ok=True)

    robot_states = pd.read_csv(STATES_PATH,index_col=0)
    # print(robot_states.head())
    robot_actions = pd.read_csv(CNTRL_PATH,index_col=0)
    # print(robot_actions.head())
    
    
    robot_state_dim = robot_states.shape[1] 
    nb_state_samples = robot_states.shape[0] 
    
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
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('data ready')
    

    n_flows = 25  # Using more flows for flexibility.
    model = ConditionalNormalizingFlow(condition_dim=robot_state_dim, n_flows=n_flows, latent_dim=robot_action_dim)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    #TODO: early stopping to prevent too much overfitting
    train_losses, val_losses = train_cnf_with_val(model=model, train_loader=train_loader, lr=1e-3, num_epochs=200, val_split=0.2)
   
    # Plot training loss.
    plt.figure()
    plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='-.')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=-100, top=100)
    plt.legend(loc='best')
    plt.title("Training Loss")
    plt.tight_layout()
    # plt.savefig(f'Data/Plots/{model_name}_Loss.svg')
    plt.show()
    

    # os.makedirs('Data/Models/Expert/CNF', exist_ok=True)
    # torch.save(model.state_dict(), f"Data/Models/Expert/CNF/{model_name}.pth")

    # test_dataset = TensorDataset(test_states)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        base_mean, _ = model.conditional_base(test_states.to(device))
        actions_pred,_ = model.inverse(base_mean, test_states.to(device))
        actions_pred = actions_pred.cpu().numpy()
        actions_pred = np.clip(actions_pred, a_min=-1.0, a_max=1.0)  # Clip actions to the valid range
        val_states_np = test_states.cpu().numpy()
        val_actions_np = test_actions.cpu().numpy()

    print(actions_pred.shape)
    print(val_states_np.shape)
    print(val_actions_np.shape)

    n_row = int(np.ceil(robot_action_dim/2))

    fig, ax = plt.subplots(nrows=n_row, ncols=2, figsize=(10,10))
    ax = ax.flatten()

    for i in range(robot_action_dim):

        ax[i].scatter(val_actions_np[:,i], actions_pred[:,i])
        ax[i].set_xlabel(f'true action dim{i}')
        ax[i].set_ylabel(f'predicted action dim {i}')
        ax[i].grid(True)
        x_min, x_max = ax[i].get_xlim()
        x=np.linspace(x_min,x_max,100)
        ax[i].plot(x,x,color='k',label='x=y')
        ax[i].legend(loc='best')
    
    plt.tight_layout()
    plt.show()

    print(f'first predicted action: {actions_pred[0,:]}')
    print(f'True action from val set: {val_actions_np[0,:]}')
