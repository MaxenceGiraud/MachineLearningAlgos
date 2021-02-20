import numpy as np
import torch
from torch import nn

class ActorNetwork(nn.Module):
    """
     This network represents the policy
    """

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.n_actions = action_size
        self.dim_observation = input_size
        
        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.n_actions),
            nn.Softmax(dim=-1)
        )
        
    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float)
        return self.net(state)
    
    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.multinomial(self.policy(state), 1)
        return action.item()

class ValueNetwork(nn.Module):
  """
   This class represents the value function
  """

  def __init__(self, input_size, hidden_size, output_size):
      super(ValueNetwork, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
      out = F.relu(self.fc1(x))
      out = F.relu(self.fc2(out))
      out = self.fc3(out)
      return out
  
  def value(self, state):
      state = torch.tensor(state, dtype=torch.float)
      return self.forward(state)

def AdvantageActorCritic(env,actor_network,value_network,actor_network_optimizer,value_network_optimizer,max_iter=200):
    ''' Advantage Actor Critic Algorithm

    Parameters
    ----------
    env  : gym-like environment,,
        Environment
    actor_network : neural network,
        Actor Network
    value_network : neural network,
        Value Network
    max_iter : int,
        Maximum number of iteration

    '''

    for iteration in range(max_iter):
        # Initialize batch storage
        batch_losses = torch.zeros(batch_size)
        batch_returns = np.zeros(batch_size)

        states = np.empty((batch_size,) + env.observation_space.shape, dtype=np.float) # shape (batch_size, state_dim)
        rewards = np.empty((batch_size,), dtype=np.float)  # shape (batch_size, )                                 
        next_states = np.empty((batch_size,) + env.observation_space.shape, dtype=np.float) # shape (batch_size, state_dim)
        dones = np.empty((batch_size,), dtype=np.bool)   # shape (batch_size, ) 
        proba = torch.empty((batch_size,), dtype=np.float)   # shape (batch_size, ), store pi(a_t|s_t)
        next_value = 0  
    
        # Intialize environment
        state = env.reset()

    # Generate batch
        for i in range(batch_size):
            action = actor_network.sample_action(state)
            next_state, reward, done, _ = env.step(action)

            states[i] = state
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
            proba[i] = actor_network.policy(state)[action]

            state = next_state
            if done:
            state = env.reset()

        if not done:
            next_value = value_network.value(next_states[-1]).detach().numpy()[0]

        # compute returns
        returns = np.zeros((batch_size,), dtype=np.float)
        T = batch_size
        for j in range(T):
            returns[T-j-1] = rewards[T-j-1]
            if j > 0:
                returns[T-j-1] += gamma * returns[T-j] * (1 - dones[T-j])
            else:
                returns[T-j-1] += gamma * next_value

        # compute advantage
        values = value_network.value(states)
        advantages = returns - values.detach().numpy().squeeze()

        # Compute MSE
        value_network_optimizer.zero_grad()
        loss_value = F.mse_loss(values, torch.tensor(returns, dtype=torch.float).unsqueeze(1)) 
        loss_value.backward()
        value_network_optimizer.step()

        # compute entropy term
        dist = actor_network.policy(states)
        entropy_term = -(dist*dist.log()).sum(-1).mean()

        # Compute Actor Gradient
        actor_network_optimizer.zero_grad()
        loss_policy = -torch.mean(torch.log(proba) * torch.tensor(advantages, dtype=torch.float))
        loss_policy += -alpha * entropy_term
        loss_policy.backward()
        actor_network_optimizer.step()

        if( (iteration+1)%10 == 0 ):
            eval_rewards = np.zeros(5)
            for sim in range(5):
                eval_done = False
                eval_state = eval_env.reset()
                while not eval_done:
                    eval_action = actor_network.sample_action(eval_state)
                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_rewards[sim] += eval_reward
                    eval_state = eval_next_state
            print("Iteration = {}, loss_value = {:0.3f}, loss_policy = {:0.3f}, rewards = {:0.2f}"
                .format(iteration +1, loss_value.item(), loss_policy.item(), eval_rewards.mean()))
            if (eval_rewards.mean() > reward_threshold):
                break
        
    return actor_network,value_network,env