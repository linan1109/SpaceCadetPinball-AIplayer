import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
import warnings
from typing import Union
from torch.nn import functional as F
import random
from collections import deque

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ReplayBuffer():
    def __init__(self, min_size, max_size, device):
        self.buffer = deque(maxlen=max_size)
        self.device = device
        self.min_size = min_size

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a.item()])
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        s_batch = torch.tensor(s_lst, dtype=torch.float, device = self.device)
        a_batch = torch.tensor(a_lst, dtype=torch.float, device = self.device)
        r_batch = torch.tensor(r_lst, dtype=torch.float, device = self.device)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float, device = self.device)

        # Normalize rewards
        r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch

    def size(self):
        return len(self.buffer)

    def start_training(self):
        # Training starts when the buffer collected enough training data.
        return self.size() >= self.min_size

class NeuralNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, hidden_activation,
                 output_activation):
        super(NeuralNetwork, self).__init__()
        self._nn = nn.Sequential()
        self._nn.add_module('input', nn.Linear(input_dim, hidden_size))
        self._nn.add_module('input_activation', hidden_activation)
        for i in range(hidden_layers):
            self._nn.add_module('hidden_{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self._nn.add_module('hidden_activation_{}'.format(i), hidden_activation)
        self._nn.add_module('output', nn.Linear(hidden_size, output_dim))
        self._nn.add_module('output_activation', output_activation)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        output = self._nn(s)
        return output


class Actor:
    def __init__(self, hidden_size: int = 256, hidden_layers: int = 2, actor_lr: float = 0.001,
                 state_dim: int = 4, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        self._net = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers - 1, nn.ReLU(), nn.Tanh())
        self._target_net = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers - 1, nn.ReLU(),
                                         nn.Tanh())
        self.optimizer = optim.Adam(self._net.parameters(), lr=self.actor_lr)

        self._log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        self.optimizer_log_std = optim.Adam([self._log_std], lr=self.actor_lr)

    def get_log_std(self) -> torch.Tensor:
        return torch.clamp(self._log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)


class Critic:
    def __init__(self, hidden_size: int = 256,
                 hidden_layers: int = 2, critic_lr: int = 0.001, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        self._net = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers - 1,
                                  nn.ReLU(), nn.Identity())
        self._target_net = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers - 1,
                                         nn.ReLU(), nn.Identity())
        self.optimizer = optim.Adam(self._net.parameters(), lr=self.critic_lr)


class TrainableParameter:

    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training,
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        self._actor = Actor(actor_lr=0.00005, device=self.device)
        self._critic_1 = Critic(device=self.device)
        self._critic_2 = Critic(device=self.device)
        self._gamma = 0.99
        self._tau = 0.001
        self._alpha = TrainableParameter(0.2, 0.0003, True, self.device)
        self._target_entropy = -1 * self.action_dim

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:

        if train:
            # sample from policy
            action_output = self._actor._net(torch.tensor(s, dtype=torch.float, device=self.device))
            action = Normal(action_output, torch.exp(self._actor.get_log_std())).sample()
            log_prob = Normal(action_output, torch.exp(self._actor.get_log_std())).log_prob(action)
            action = action.cpu().detach().numpy()
            action = action.reshape(1, )
            action = np.clip(action, -1, 1)

        else:
            action = self._actor._target_net(torch.tensor(s, dtype=torch.float, device=self.device))
            log_prob = Normal(action, torch.exp(self._actor.get_log_std())).log_prob(action)
            action = action.cpu().detach().numpy()
            action = action.reshape(1, )

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action

    def get_action_prob(self, s: np.ndarray, train: bool) -> np.ndarray:


        if train:
            # sample from policy
            action_output = self._actor._net(torch.tensor(s, dtype=torch.float, device=self.device))
            action = Normal(action_output, torch.exp(self._actor.get_log_std())).sample()
            log_prob = Normal(action_output, torch.exp(self._actor.get_log_std())).log_prob(action)
            action = action.cpu().detach().numpy()
            action = np.clip(action, -1, 1)

        else:
            action = self._actor._target_net(torch.tensor(s, dtype=torch.float, device=self.device))
            log_prob = Normal(action, torch.exp(self._actor.get_log_std())).log_prob(action)
            action = action.cpu().detach().numpy()

        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'
        return action, log_prob

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):

        # Batch sampling
        for i in range(50):
            batch = self.memory.sample(self.batch_size)
            s_batch, a_batch, r_batch, s_prime_batch = batch

            _, log_prob = self.get_action_prob(s_prime_batch, True)
            Q_prime_1 = self._critic_1._target_net(
                torch.cat((s_prime_batch, self._actor._target_net(s_prime_batch)), dim=1))
            Q_prime_2 = self._critic_2._target_net(
                torch.cat((s_prime_batch, self._actor._target_net(s_prime_batch)), dim=1))
            Q_prime = torch.min(Q_prime_1, Q_prime_2)
            y = r_batch + self._gamma * (Q_prime - self._alpha.get_param() * log_prob)

            # Critics update here.
            Q_a = self._critic_1._net(torch.cat((s_batch, a_batch), dim=1))
            loss_critic = F.mse_loss(Q_a, y.detach())
            self.run_gradient_update_step(self._critic_1, loss_critic)
            Q_a = self._critic_2._net(torch.cat((s_batch, a_batch), dim=1))
            loss_critic = F.mse_loss(Q_a, y.detach())
            self.run_gradient_update_step(self._critic_2, loss_critic)

            # Policy update here
            Q_pai_1 = self._critic_1._net(torch.cat((s_batch, self._actor._net(s_batch)), dim=1))
            Q_pai_2 = self._critic_2._net(torch.cat((s_batch, self._actor._net(s_batch)), dim=1))
            Q_pai = torch.min(Q_pai_1, Q_pai_2)
            actions, log_prob = self.get_action_prob(s_batch, True)
            loss_actor = (self._alpha.get_param() * log_prob - Q_pai).mean()
            self.run_gradient_update_step(self._actor, loss_actor)

            # update log_std for actor
            # log_prob = Normal(actions, torch.exp(self._actor.get_log_std())).log_prob(a_batch)
            actions, log_prob = self.get_action_prob(s_batch, True)
            loss_log_std = -log_prob.mean()
            self._actor.optimizer_log_std.zero_grad()
            loss_log_std.backward()
            self._actor.optimizer_log_std.step()

            # update alpha
            alpha_loss = -(self._alpha.get_param() * (log_prob + self._target_entropy).detach()).mean()
            self._alpha.optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha.optimizer.step()

            # update target network
            self.critic_target_update(self._critic_1._net, self._critic_1._target_net, self._tau, True)
            self.critic_target_update(self._critic_2._net, self._critic_2._target_net, self._tau, True)
            self.critic_target_update(self._actor._net, self._actor._target_net, self._tau, True)
