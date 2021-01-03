import numpy as np
from copy import deepcopy
from ..dl.optimizer import Adam


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        r= np.random.choice(np.arange(len(self.memory)), batch_size)
        return np.array(self.memory)[r].tolist()

    def __len__(self):
        return len(self.memory)


class DeepQLearning:
    ''' Deep Q learning 

    Parameters
    ----------
    n_episode : int,
        Number of episodes to train on
    buffer_capacity : int,
        Capacity of the Replay Buffer
    batch_size : int,
        batch size to train the nn with
    gamma : float,
        Discount factor
    epsilon : float,
        ...
    eval_every: int,
        Evaluate nn every 'eval_every' steps
    reward_threshold : int,
        Maximum value of reward, if receive this reward -> stops
    update_target_every : int,
        Number of steps the target network is updated
    '''
    def __init__(self,n_episode=500,buffer_capacity=10000,batch_size = 256,gamma =0.99,epsilon = 0.99,eval_every=5,reward_threshold=200,update_target_every=20):
        self.n_episode = n_episode
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_every = eval_every
        self.reward_threshold = reward_threshold
        self.update_target_every = update_target_every

    
    def _get_q(self,states):
        return self.nn.predict(states)
    
    def _choose_action(self,state, epsilon):
        ''' Return action according to an epsilon-greedy exploration policy '''
        if np.random.uniform() < epsilon:
            return self.env.action_space.sample()
        else:
            q = self._get_q([state])
            return q.argmax() 
    
    def _eval_dqn(self,n_sim=5):
        ''' Monte Carlo evaluation of DQN agent.

        Repeat n_sim times:
            * Run the DQN policy until the environment reaches a terminal state (= one episode)
            * Compute the sum of rewards in this episode
            * Store the sum of rewards in the episode_rewards array.
        '''
        env_copy = deepcopy(self.env)
        episode_rewards = np.zeros(n_sim)

        for ii in range(n_sim):
            state = env_copy.reset()
            done = False 
            while not done:
                action = self._choose_action(state, 0.0)
                next_state, reward, done, _ = env_copy.step(action)
                episode_rewards[ii] += reward
                state = next_state
        return episode_rewards


    def _update(self,state, action, reward, next_state, done,ep):
    
        # add data to replay buffer
        if done:
            next_state = None
        self.replay_buffer.push(state, action, reward, next_state)
        
        if len(self.replay_buffer) < self.batch_size:
            return np.inf
        
        # get batch
        transitions = self.replay_buffer.sample(self.batch_size)

        # process batch of (state, action, reward, next_state)
        states = [transitions[ii][0] for  ii in range(self.batch_size)]
        actions = [transitions[ii][1] for  ii in range(self.batch_size)] 
        rewards = [transitions[ii][2] for  ii in range(self.batch_size)] 

        # Attention: next_state is None when the previous state is terminal.
        # we handle this using a mask.
        next_states = [transitions[ii][3] for  ii in range(self.batch_size) if transitions[ii][3] is not None ] 
        mask = [transitions[ii][3] is not None for  ii in range(self.batch_size)]

        # Q(s_i, a_i)
        values = self.nn.predict(states) # TODO use actions
        # values = torch.gather(values, dim=1, index=actions_torch)

        # max_a Q(s_{i+1}, a)
        values_next_states = np.zeros(self.batch_size)
        values_next_states[mask] = self.target_network.predict(next_states).max(axis=1)
        # values_next_states = values_next_states.view(-1, 1) # ??? reshape ??

        # targets y_i
        targets = rewards + self.gamma*values_next_states
        

        # Loss function / forward pass
        targets = np.array([targets,targets]).T# tmp solution
        loss = self.nn.forward(np.array(states), targets)
        
        # Optimize the model  / Backprop
        self.nn.backprop(targets)
        self.optimizer.update(nn=self.nn,t=ep)
        
        return loss

    def train(self,nn,env,optimizer=Adam(learning_rate=0.1)):
        '''
        Parameters
        -----------
        nn : Neural netwok,
            neural network to train, need to have a predit, forward and backprop method
        env : gym-like environment,
            environment to train on
        optimizer : optimizer,
            Use to optimize the nn, must have an update method that update the weights of the nn
        '''
        self.nn = nn
        self.target_network = deepcopy(self.nn)
        self.env = env
        self.optimizer = optimizer

        state = self.env.reset()
        ep = 0
        total_time = 0
        if hasattr(self.optimizer,'init_layers'):
            self.optimizer.init_layers(self.nn)
        while ep < self.n_episode:
            action = self._choose_action(state, self.epsilon)

            # take action and update replay buffer and networks
            next_state, reward, done, _ = self.env.step(action)
            _ = self._update(state, action, reward, next_state, done,ep)

            # update state
            state = next_state

            # end episode if done
            if done:
                state = self.env.reset()
                ep   += 1
                if ( (ep+1)% self.eval_every == 0):
                    rewards = self._eval_dqn()
                    print("episode =", ep+1, ", reward = ", np.mean(rewards))
                    if np.mean(rewards) >= self.reward_threshold:
                        break

                # update target network
                if ep % self.update_target_every == 0:
                    self.target_network = deepcopy(self.nn)
            
            total_time += 1

        if hasattr(self.nn,'clear_layer_training'):
            self.nn.clear_layer_training(self.nn)