import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

from agents.td_n import *

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), 'results', 'TDN')
os.makedirs(results_dir, exist_ok=True)

# Configuration
training_steps = 15000
td_orders = [1, 2, 3, 4, 5]  # Different TD orders to try
gamma = 0.99
buffer_length = 1000
batch_size = 100
warm_up = 0
NOISE = 'Gaussian'  # 'Gaussian' or 'OrnsteinUhlenbeck'

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = env.action_space.low[0]
action_high = env.action_space.high[0]

# Store results from each TD order
all_results = {}

for td_order in td_orders:
    print(f"\nTraining with TD order: {td_order}")
    BEST_SO_FAR = -np.inf
    
    if NOISE == 'Gaussian':
        noise = Normal(loc=0, scale=0.2)
    elif NOISE == 'OrnsteinUhlenbeck':
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.1)
    else:
        raise ValueError('Invalid noise type. Choose "Gaussian" or "OrnsteinUhlenbeck".')
    
    behavior_policy = Policy(state_dim=state_dim, action_dim=action_dim)
    target_policy = Policy(state_dim=state_dim, action_dim=action_dim)
    behavior_q = Value(state_dim=state_dim, action_dim=action_dim)
    target_q = Value(state_dim=state_dim, action_dim=action_dim)
    
    # New behaviorPolicy and behaviorQ network weights for each cycle: N(0, 0.1)
    init_model_weights(behavior_policy)
    init_model_weights(behavior_q)
    
    # PolicyWeights_t <-- PolicyWeights_b | QWeights_t <-- QWeights_b
    target_policy.load_state_dict(behavior_policy.state_dict())
    target_q.load_state_dict(behavior_q.state_dict())
    
    agent = TDN(
        order=td_order,
        policy_network=behavior_policy,
        target_policy=target_policy,
        value_network=behavior_q,
        target_value_function=target_q,
        discount_factor=gamma)
    
    memory = TDNMemory(order=td_order, state_dim=state_dim, action_dim=action_dim, buffer_length=buffer_length)
    
    obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
    episodic_returns = []
    cumulative_reward = 0
    
    list_of_all_the_data = []
    
    progress_bar = tqdm(range(training_steps), unit="step")
    for t in progress_bar:
        if t < warm_up:
            clipped_action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = behavior_policy.forward(torch.tensor(obs, dtype=torch.float32, device='cpu'))
                expl_noise = noise.sample(action.shape).cpu().numpy()
            noisy_action = action.cpu().numpy() + expl_noise
            clipped_action = np.clip(noisy_action, a_min=action_low, a_max=action_high)
    
        obs_, reward, terminated, truncated, _ = env.step(clipped_action)
        
        done = terminated or truncated
    
        cumulative_reward += reward
        memory.add_sample(state=obs, action=clipped_action, reward=reward, next_state=obs_, done=done)
    
        if (t >= warm_up) and (memory.size >= batch_size):
            agent.train(memory_buffer=memory, batch_size=batch_size, epochs=1)
    
        if done:
            episodic_returns.append(cumulative_reward)
            if cumulative_reward > BEST_SO_FAR:
                BEST_SO_FAR = cumulative_reward
                # Save model to results directory
                model_path = os.path.join(results_dir, f'best_policy_td{td_order}.pth')
                torch.save(behavior_policy.state_dict(), model_path)
    
            cumulative_reward = 0
            obs, _ = env.reset(options={'x_init': np.pi, 'y_init': 8.0})
        else:
            obs = obs_.copy()
    
    # Collect stats
    for i in range(len(agent.pi_loss)):
        list_of_all_the_data.append({
            'step': i,
            'policy_loss': agent.pi_loss[i],
            'q_loss': agent.q_loss[i],
            'return': episodic_returns[i] if i < len(episodic_returns) else np.nan,
        })
    
    # Save results to results directory
    df = pd.DataFrame(list_of_all_the_data)
    csv_path = os.path.join(results_dir, f"data_td{td_order}.csv")
    df.to_csv(csv_path, index=False)
    
    # Store returns for comparison plot
    all_results[f'TD-{td_order}'] = episodic_returns

env.close()

# Create comparison plot
plt.figure(figsize=(12, 8))
for name, returns in all_results.items():
    plt.plot(returns, alpha=0.3, label=f'{name} (Raw)')

    plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('Comparison of Episodic Returns for Different TD Orders')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to results directory
plot_path = os.path.join(results_dir, 'td_order_comparison.png')
plt.savefig(plot_path)
plt.show()

print(f"Training complete! Results saved to {results_dir}")
