from typing import Literal

import torch
import torch.nn as nn

import icfalstm.model.layer as mylayer

__all__ = ['RNNBase']


class RNNBase(nn.Module):
    """Base class for RNN models.

    Attributes:
        mode (str): The mode of the RNN.
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        hidden_size (int): The hidden size.
        num_outputs (int): The number of outputs.
        device (torch.device): The device to use.
        batch_first (bool): Whether the batch is the first dimension.
        prelayer (torch.nn.Module): The prelayer.
        prelayer_mode (str): The mode of the prelayer.
        layer (torch.nn.Module): The layer.
        layer_mode (str): The mode of the layer.
        norm (torch.nn.Module): The normalization layer.
        inverse_norm (torch.nn.Module): The inverse normalization layer.
        output_layer (torch.nn.Module): The output layer.
    """

    def __init__(self,
                 mode: str,
                 input_size: tuple[int, int],
                 hidden_size: int,
                 num_outputs: int,
                 device: torch.device,
                 batch_first: bool = True) -> None:
        """Initializes the RNNBase.

        Args:
            mode (str): The mode of the RNN.
            input_size (tuple[int, int]): The input size.
            hidden_size (int): The hidden size.
            num_outputs (int): The number of outputs.
            device (torch.device): The device to use.
            batch_first (bool, optional): Whether the batch is the first. 
                Defaults to True.
        """
        super(RNNBase, self).__init__()
        self.mode = mode
        self.num_cities, self.num_attrs = input_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.device = device
        self.batch_first = batch_first
        self._init_network()

    def _init_network(self):
        """Initializes the network.
        """
        self.prelayer = None
        if '-' in self.mode:
            self.prelayer_mode, self.layer_mode = self.mode.split('-')
            self.prelayer = getattr(mylayer, self.prelayer_mode)(
                input_size=(self.num_cities, self.num_attrs),
                device=self.device)
        else:
            self.layer_mode = self.mode
        self.layer = getattr(mylayer, f'{self.layer_mode}Cell')(
            input_size=(self.num_cities, self.num_attrs),
            num_hiddens=self.hidden_size,
            device=self.device)
        self.norm = mylayer.MaxMinNorm()
        self.inverse_norm = mylayer.InverseMaxMinNorm()
        self.output_layer = nn.Linear(self.hidden_size, self.num_outputs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): The input.

        Returns:
            torch.Tensor: The output.
        """
        x = input.permute(1, 0, 2, 3) if self.batch_first else input
        output = []
        state = None
        for time_x in x:
            y, max_vals = self.norm(time_x)
            if self.prelayer is not None:
                y = self.prelayer(y)
            y, state = self.layer(y, state)
            y = self.inverse_norm(y, max_vals)
            output.append(self.output_layer(y).unsqueeze(0))
        return torch.cat(output, dim=0).permute(1, 0, 2, 3)