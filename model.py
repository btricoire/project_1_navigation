import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers_units=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layers_units(list of int):
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.ModuleList()
        prev_unit = state_size
        for i, units in enumerate(layers_units):
            self.fc.append(nn.Linear(prev_unit, units))
            prev_unit = units

        self.fc_final = nn.Linear(prev_unit, action_size)

    def forward(self, state):
        x = state
        for fct_cur in self.fc:
            x = F.relu(fct_cur(x))
        return self.fc_final(x)
