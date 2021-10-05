import torch
from torch import nn
import copy
from collections import deque
import random

class DQN_Agent:

    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_fc_net_tanh(layer_sizes).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().to(device)
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def build_fc_net_tanh(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def load_pretrained_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_net.state_dict(), model_path)

    def get_action(self, state, action_space_len, epsilon):
        assert state.ndim == 1, "get_action expects to return the action for a single state, not a vector of states"
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, action_space_len, (1,))
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
        self.experience_replay.append(experience)
        return

    def sample_from_buffer(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).long()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        self.network_sync_counter += 1

        states, actions, rewards, next_states = self.sample_from_buffer(sample_size=batch_size)

        # predict expected return of current state using main network
        qs = self.q_net(states.to(device))   # ToDo: can these be returned from the buffer as to(device)?
        # pred_return, _ = torch.max(qs, axis=1)
        pred_return = qs.gather(index=actions.unsqueeze(-1), dim=1).squeeze(1)

        # get target return using target network
        q_next = self.get_bootstrapping_values(next_states.to(device))
        target_return = rewards.to(device) + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()


import gym
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    exp_replay_size = 256
    agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                      exp_replay_size=exp_replay_size)

    # Main training loop
    losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
    episodes = 10000
    epsilon = 1

    # initialize experience replay
    index = 0
    for i in range(exp_replay_size):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs, env.action_space.n, epsilon=1)
            obs_next, reward, done, _ = env.step(action)
            agent.add_to_buffer([obs, action, reward, obs_next])
            obs = obs_next
            index += 1
            if index > exp_replay_size:
                break

    index = 128
    for i in tqdm(range(episodes)):
        obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
        while not done:
            ep_len += 1
            action = agent.get_action(obs, env.action_space.n, epsilon)
            obs_next, reward, done, _ = env.step(action)
            agent.add_to_buffer([obs, action, reward, obs_next])

            obs = obs_next
            rew += reward
            index += 1

            if index > 128:
                index = 0
                for j in range(4):
                    loss = agent.train(batch_size=16)
                    losses += loss
        if epsilon > 0.05:
            epsilon -= (1 / 5000)

        losses_list.append(losses / ep_len), reward_list.append(rew)
        episode_len_list.append(ep_len), epsilon_list.append(epsilon)

    print("Saving trained model")
    agent.save_trained_model("cartpole-dqn.pth")


def test():

    env = gym.make('CartPole-v0')
    # env = gym.wrappers.Monitor(env, "record_dir", force='True')

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    exp_replay_size = 256
    agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                      exp_replay_size=exp_replay_size)
    agent.load_pretrained_model("cartpole-dqn.pth")

    reward_arr = []
    for i in tqdm(range(100)):
        obs, done, rew = env.reset(), False, 0
        while not done:
            action = agent.get_action(obs, env.action_space.n, epsilon=0)
            obs, reward, done, info = env.step(action)
            rew += reward
            time.sleep(0.01)
            env.render()

        reward_arr.append(rew)
    print("average reward per episode :", sum(reward_arr) / len(reward_arr))


if __name__ == '__main__':
    train()
    test()