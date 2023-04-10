from typing import Literal

import torch
import torch.nn as nn

import icfalstm.nn.layer as mylayer

__all__ = ['RNNBase']


class RNNBase(nn.Module):
    """Base class for RNN models.

    Attributes:
        mode (['ICFA-LSTM', 'ICA-LSTM', 'LSTM', 'ICFA-GRU', 'ICA-GRU', 'GRU', 
            'ICFA-RNN', 'ICA-RNN', 'RNN']): RNN mode.
        map_units (int): Number of input units.
        num_attrs (int): Number of attributes.
        hidden_units (int): Number of hidden units.
        num_outputs (int): Number of output units.
        device (torch.device): Device to use.
        batch_first (bool): Whether the first dimension of the input is the 
            batch size. Default: True.
        norm (MaxMinNorm): Normalization layer.
        assoc_layer (ICFA or ICA): Association layer.
        body_layer (LSTMCell, GRUCell, or RNNCell): Body layer.
        output_layer (Dense): Output layer.
    """

    def __init__(self,
                 mode: Literal['ICFA-LSTM', 'ICA-LSTM', 'LSTM', 'ICFA-GRU',
                               'ICA-GRU', 'GRU', 'ICFA-RNN', 'ICA-RNN', 'RNN'],
                 map_units: int,
                 num_attrs: int,
                 hidden_units: int,
                 num_outputs: int,
                 device: torch.device,
                 batch_first: bool = True) -> None:
        """Initializes the RNNBase class.

        Args:
            mode (['ICFA-LSTM', 'ICA-LSTM', 'LSTM', 'ICFA-GRU', 'ICA-GRU', 
                'GRU', 'ICFA-RNN', 'ICA-RNN', 'RNN']): The RNN mode.
            map_units (int): The number of input units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            num_outputs (int): The number of output units.
            device (torch.device): The device to use.
            batch_first (bool, optional): Whether the first dimension of the 
                input is the batch size. Defaults to True.
        """
        super(RNNBase, self).__init__()
        self.mode = mode
        self.map_units = map_units
        self.num_attrs = num_attrs
        self.hidden_units = hidden_units
        self.num_outputs = num_outputs
        self.device = device
        self.batch_first = batch_first
        self._init_network()

    def _init_network(self):
        """Initializes the network."""
        self.norm = mylayer.MaxMinNorm()
        self.assoc_layer = None
        if '-' in self.mode:
            assoc_layer_mode, body_layer_mode = self.mode.split('-', 1)
            self.assoc_layer = getattr(mylayer,
                                       assoc_layer_mode)(self.map_units,
                                                         self.num_attrs,
                                                         self.hidden_units,
                                                         self.device)
            self.body_layer = getattr(mylayer, f'{body_layer_mode}Cell')(
                self.map_units, self.hidden_units, self.hidden_units,
                self.device)
        else:
            self.body_layer = getattr(mylayer,
                                      f'{self.mode}Cell')(self.map_units,
                                                          self.hidden_units,
                                                          self.device)
        self.output_layer = mylayer.Dense(self.map_units, self.hidden_units,
                                          self.num_outputs, self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.transpose(input, 0, 1) if self.batch_first else input
        output = []
        state = None
        for time_x in x:
            y = self.norm(time_x)
            if self.assoc_layer is not None:
                y = self.assoc_layer(y)
            state = self.body_layer(y, state)
            y = self.norm(y, inverse=True)
            y = self.output_layer(y)
            y = torch.unsqueeze(y, 0)
            output.append(y)
        output = torch.cat(output, 0)
        output = torch.transpose(output, 0, 1) if self.batch_first else output
        return output.squeeze(-1)
