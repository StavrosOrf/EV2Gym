import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output Layer
        self.mu = nn.Linear(hidden_size[1], num_outputs)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V
