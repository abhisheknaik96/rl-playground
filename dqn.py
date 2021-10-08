### Based on https://github.com/mahakal001/reinforcement-learning/tree/master/cartpole-dqn

import torch
from torch import nn
import copy
from collections import deque
import random
import gym
from tqdm import tqdm
import time
import numpy as np


def get_stats(results):
    mean = np.round(np.mean(results), 2)
    std = np.round(np.std(results), 2)
    min = np.round(np.min(results), 2)
    max = np.round(np.max(results), 2)
    median = np.round(np.median(results), 2)
    return mean, std, min, max, median


def build_fc_net_tanh(layer_sizes):
    assert len(layer_sizes) > 1
    layers = []
    for index in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
        act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
        layers += (linear, act)
    return nn.Sequential(*layers)


class DQNAgent:

    def __init__(self, config):

        self.num_actions = config['num_actions']
        self.num_features = config['num_features']

        self.seed = None
        self.rng = None
        self.gamma = None
        self.epsilon = None
        self.layer_sizes = None
        self.q_net = None
        self.target_net = None
        self.net_sync_freq = None
        self.net_sync_counter = None
        self.load_model_from = None
        self.param_update_freq = None
        self.step_size = None
        self.buffer_size = None
        self.experience_buffer = None
        self.loss_fn = None
        self.optimizer = None
        self.batch_size = None
        self.device = None
        self.last_obs = None
        self.last_action = None
        self.counter = None
        self.save_model = None
        self.save_model_loc = None

    def agent_init(self, agent_info):

        self.seed = agent_info.get('rng_seed', 42)
        # self.rng = torch.Generator()
        # self.rng.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)  # ToDo: don't like this. Remove need for this package

        self.gamma = torch.tensor(agent_info.get('gamma', 0.95)).float().to(device)
        self.epsilon = torch.tensor(agent_info.get('epsilon', 0.2)).float().to(device)

        assert 'layer_sizes' in agent_info, "layer_sizes needs to be specified in agent_info"
        self.layer_sizes = agent_info['layer_sizes']

        assert 'device' in agent_info, "device needs to be specified in agent_info"
        self.device = agent_info['device']

        self.q_net = build_fc_net_tanh(self.layer_sizes).to(self.device)
        self.load_model_from = agent_info.get('load_model_from', None)
        if self.load_model_from:
            self.q_net.load_state_dict(torch.load(self.load_model_from))
            print(f'Successfully loaded model from {self.load_model_from}')
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.net_sync_freq = agent_info.get('net_sync_freq', 256)
        self.net_sync_counter = 0
        self.param_update_freq = agent_info.get('param_update_freq', 32)

        self.step_size = agent_info.get('step_size', 1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.step_size)
        self.batch_size = agent_info.get('batch_size', 32)

        self.buffer_size = agent_info.get('buffer_size', 50000)
        self.experience_buffer = deque(maxlen=self.buffer_size)

        self.counter = 0
        self.save_model = agent_info.get('save_model', False)
        self.save_model_loc = agent_info.get('save_model_loc', 'models/cartpole-dqn')

    def save_trained_model(self, suffix):
        filename = self.save_model_loc + suffix + '.pth'
        torch.save(self.q_net.state_dict(), filename)

    def choose_action(self, state):
        assert state.ndim == 1, "get_action expects to return the action for a single state, not a vector of states"
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.num_actions, (1,))
        else:
            with torch.no_grad():   # because gradients not required here
                qs = self.q_net(torch.from_numpy(state).float().to(device))
            _, action = torch.max(qs, axis=0)
            ### ToDo: uhh, torch doesn't have an obvious way to get all argmax elements and then choose one randomly
            # action_idx = torch.randint(0, actions.shape[0]+1, (1,))
            # action = actions[action_idx]
        return action.item()        # return the actual integer instead of the tensor

    def get_bootstrapping_values(self, next_state_vec):
        with torch.no_grad():
            qs_next = self.target_net(next_state_vec)     # ToDo: check why this doesn't need to(device)
        q_next, _ = torch.max(qs_next, axis=1)
        return q_next

    def add_to_buffer(self, experience):
        self.experience_buffer.append(experience)
        return

    def sample_from_buffer(self):

        num_samples = self.buffer_size if len(self.experience_buffer) >= self.buffer_size else len(self.experience_buffer)
        sample = random.sample(self.experience_buffer, num_samples)
        states = torch.tensor([exp[0] for exp in sample]).float()    # ToDo: check if all can be sent to device here itself
        actions = torch.tensor([exp[1] for exp in sample]).long()
        rewards = torch.tensor([exp[2] for exp in sample]).float()
        next_states = torch.tensor([exp[3] for exp in sample]).float()
        return states, actions, rewards, next_states

    def update_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon -= (1 / 5000)

    def process_raw_observation(self, next_state):
        return next_state

    def agent_start(self, first_state):
        observation = self.process_raw_observation(first_state)
        action = self.choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action

    def agent_step(self, reward, next_state, info):
        self.counter += 1
        loss = None

        observation = self.process_raw_observation(next_state)
        self.add_to_buffer([self.last_obs, self.last_action, reward, observation])

        # if time to update parameters
        if self.counter % self.param_update_freq == 0:
            # if time to update target network
            if self.counter % self.net_sync_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            # update q_net parameters
            loss = self.update_params()

        self.update_epsilon()
        action = self.choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action, loss

    def update_params(self):

        # sample a batch of transitions
        states, actions, rewards, next_states = self.sample_from_buffer()

        # predict expected return of current state using main network
        qs = self.q_net(states.to(device))  # ToDo: can these be returned from the buffer as to(device)?
        # pred_return, _ = torch.max(qs, axis=1)
        pred_return = qs.gather(index=actions.unsqueeze(-1), dim=1).squeeze(1)

        # get target return using target network
        q_next = self.get_bootstrapping_values(next_states.to(device))
        target_return = rewards.to(device) + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # ToDo: I don't think retain_graph should be True
        self.optimizer.step()

        return loss.item()

    # def close(self):



def experiment_loop(exp_info, env_info, agent_info):
    # Main training loop
    # losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
    num_runs = exp_info.get('num_runs', 5)
    num_episodes = exp_info.get('num_episodes', 10000)
    returns_all = np.zeros((num_runs, num_episodes))
    render = env_info.get('render', False)

    for run in range(num_runs):

        env = gym.make('CartPole-v0')
        env.seed(run)
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DQNAgent(config={'num_features': input_dim, 'num_actions': output_dim})
        agent_info['rng_seed'] = run
        agent.agent_init(agent_info)

        for eps in tqdm(range(num_episodes)):
            obs, done, losses, eps_len, eps_ret = env.reset(), False, 0, 0, 0
            action = agent.agent_start(obs)
            if render:
                env.render()
                time.sleep(0.01)

            while not done:
                eps_len += 1
                obs_next, reward, done, _ = env.step(action)
                action, loss = agent.agent_step(reward, obs_next, False)
                eps_ret += reward
                if loss is not None:
                    losses += loss
                if render:
                    env.render()
                    time.sleep(0.01)

            # losses_list.append(losses / eps_len), reward_list.append(eps_ret)
            # episode_len_list.append(eps_len), epsilon_list.append(agent.epsilon)
            returns_all[run][eps] = eps_ret

        # print(f'Average episode length: {sum(episode_len_list)/num_episodes}')
        # print(f'Average episode length (last half of training): {sum(episode_len_list[num_episodes//2:])/(num_episodes/2)}')
        print(f'Stats (mean, std, min, max, median): {get_stats(returns_all[run])}')
        print(f'Stats (last half): {get_stats(returns_all[run, num_episodes//2:])}')
        if agent.save_model:
            print("Saving trained model")
            suffix = f'_{run}'
            agent.save_trained_model(suffix)
        env.close()

    print(f'\n\nFinal stats (mean, std, min, max, median): {get_stats(returns_all)}')
    print(f'Final stats (last 50%): {get_stats(returns_all[:, num_episodes//2:])}')
    print(f'Final stats (last 10%): {get_stats(returns_all[:, int(0.9*num_episodes):])}')


def run(save_model_loc=None, load_model_from=None, device='cpu', mode='eval'):

    if mode=='train':
        epsilon = 1.0
        step_size = 1e-3
        save_model = True
        num_runs = 5
        num_episodes = 10000
        param_update_freq = 32
        render = False
    else:
        epsilon = 0.0
        step_size = 0.0
        save_model = False
        num_runs = 1
        num_episodes = 10
        param_update_freq = 10000
        render = True

    experiment_loop(
        exp_info={'num_episodes': num_episodes,
                  'num_runs': num_runs},
        env_info={'render': render},
        agent_info={'layer_sizes': [4, 64, 2],
                    'step_size': step_size,
                    'epsilon': epsilon,
                    'rng_seed': 1423,
                    'net_sync_freq': 128,
                    'buffer_size': 256,
                    'param_update_freq': param_update_freq,
                    'batch_size': 16,
                    'device': device,
                    'save_model': save_model,
                    'save_model_loc': save_model_loc,
                    'load_model_from': load_model_from}
    )

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # run('models/cartpole-dqn', device=device, render=False, num_episodes=10000, num_runs=5, save_model=True)
    # run('models/cartpole-dqn.pth_4.pth', device=device, render=True, num_episodes=50, num_runs=1, save_model=False)

    run(save_model_loc='models/cartpole-dqn', device=device, mode='train')
    # run(load_model_from='models/cartpole-dqn_4.pth', device=device, mode='eval')
