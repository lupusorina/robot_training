from ppo import Agent
import torch
import numpy as np
from tests.cart_pole import CartPole
import matplotlib.pyplot as plt

env = CartPole()
policy_layers = [4, 64, 64, 1 * 2]

agent = Agent(policy_layers=policy_layers,
              value_layers=[4, 64, 64, 1], # Not used
              entropy_cost=0.01, # Not used
              discounting=0.97, # Not used
              reward_scaling=1.0, # Not used
              device='cpu')

agent.policy.load_state_dict(torch.load('logs/results/ppo_model_pytorch.pth', weights_only=True))
agent.eval()

obs = env.reset()
print('Start angle in degrees:', obs[1] * 180 / np.pi)
obs_list = []
for i in range(1000):
    _, action = agent.get_logits_action(torch.tensor(obs, dtype=torch.float32))
    action_np = action.cpu().detach().numpy()
    obs, _, _, _, = env.step(action_np)
    obs_list.append(obs)

obs_list = np.array(obs_list)
plt.plot(np.rad2deg(obs_list[:, 1]))
# zero horizontal line
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
    
    