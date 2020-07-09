# (c) 2019-​2020,   Emanuel Joos  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
# Based on https://github.com/higgsfield/RL-Adventure-2
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


def fanin_init(m):
    if isinstance(m, nn.Linear):
        fanin = m.weight.data.size()[0]
        v = 1.0 / (np.sqrt(fanin))
        nn.init.uniform_(m.weight, -v, v)
        nn.init.uniform_(m.bias, -v, v)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        self.action_std = 0

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        self.action_std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, self.action_std)
        return dist, value
