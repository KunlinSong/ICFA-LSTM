from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

__all__ = [
    'Dense', 'GRUCell', 'GLSTMCell', 'ICA', 'ICFA', 'InverseMaxMinNorm',
    'LSTMCell', 'MaxMinNorm'
]


def _get_param(shape: tuple[int], device: torch.device) -> Parameter:
    """Gets a parameter.

    Args:
        shape (tuple[int]): The shape of the parameter.
        device (torch.device): The device to put the parameter on.

    Returns:
        Parameter: The parameter.
    """
    return Parameter(
        torch.randn(shape, device=device, dtype=torch.float64) / 100)


def _get_bias(shape: tuple[int], device: torch.device) -> Parameter:
    """Gets a bias.

    Args:
        shape (tuple[int]): The shape of the bias.
        device (torch.device): The device to put the bias on.

    Returns:
        Parameter: The bias.
    """
    return Parameter(torch.zeros(shape, device=device, dtype=torch.float64))


def _gate_params(num_cities: int, num_attrs: int, num_hiddens: int,
                 device: torch.device) -> tuple[Parameter]:
    """Gets the parameters for the gates.

    Args:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to put the parameters on.

    Returns:
        tuple[Parameter, Parameter, Parameter]: The parameters for the gates.
    """
    return (_get_param((num_cities, num_hiddens),
                       device), _get_param((num_hiddens, num_hiddens), device),
            _get_bias((num_hiddens,), device))


class Map(nn.Module):
    """A base class for the map layers.

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        device (torch.device): The device to put the parameters on.
    """

    def __init__(self, input_size: tuple[int, int],
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (num_cities, 
                num_attrs).
            device (torch.device): The device to put the parameters on.
        """
        super(Map).__init__()
        self.num_cities, self.num_attrs = input_size
        self.device = device


class MaxMinNorm(nn.Module):
    """A class for the max-min normalization.
    """

    def __init__(self) -> None:
        super(MaxMinNorm, self).__init__()

    def forward(self,
                input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor and the 
                maximum value. 
        """
        max_vals = torch.max(torch.abs(input), dim=(1, 2)).values
        return input / max_vals, max_vals


class InverseMaxMinNorm(nn.Module):
    """A class for the inverse max-min normalization.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor,
                max_vals: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.
            max_vals (torch.Tensor): The maximum value.

        Returns:
            torch.Tensor: The output tensor.
        """
        return input * max_vals


class ICFA(Map):
    """A class for the ICFA layer.

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        device (torch.device): The device to put the parameters on.
        w_assco_[i] (Parameter): The association matrix for the i-th attribute.
    """

    def __init__(self, input_size: tuple[int, int],
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (num_cities,
                num_attrs).
            device (torch.device): The device to put the parameters on.
        """
        super(ICFA, self).__init__(input_size=input_size, device=device)
        for idx in range(self.num_attrs):
            w_assoc = _get_param((self.num_cities, self.num_cities),
                                 device=self.device)
            setattr(self, f'w_assoc_{idx}', w_assoc)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        result = []
        for idx, mat in enumerate(
                torch.split(input, split_size_or_sections=1, dim=-1)):
            w_assoc = getattr(self, f'w_assoc_{idx}')
            result.append(torch.matmul(mat.squeeze(-1), w_assoc).unsqueeze(-1))
        return torch.cat(result, dim=-1)


class ICA(Map):
    """A class for the ICA layer.

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        device (torch.device): The device to put the parameters on.
        w_assoc (Parameter): The association matrix.
    """

    def __init__(self, input_size: tuple[int, int],
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (num_cities,
                num_attrs).
            device (torch.device): The device to put the parameters on.
        """
        super(ICA, self).__init__(input_size=input_size, device=device)
        self.w_assoc = _get_param((self.num_cities, self.num_cities),
                                  device=self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = input.permute(0, 2, 1).reshape(-1, self.num_cities)
        x = torch.matmul(x, self.w_assoc)
        return x.reshape(-1, self.num_attrs, self.num_cities).permute(0, 2, 1)


class LSTMCell(Map):
    """A class for the LSTM cell. It is a LSTM cell sith a map input. The 
    input is a tensor of shape (batch_size, num_cities, num_attrs).

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to put the parameters on.
        w_i (Parameter): The input gate weight.
        u_i (Parameter): The input gate weight.
        b_i (Parameter): The input gate bias.
        w_f (Parameter): The forget gate weight.
        u_f (Parameter): The forget gate weight.
        b_f (Parameter): The forget gate bias.
        w_o (Parameter): The output gate weight.
        u_o (Parameter): The output gate weight.
        b_o (Parameter): The output gate bias.
        w_g (Parameter): The candidate gate weight.
        u_g (Parameter): The candidate gate weight.
        b_g (Parameter): The candidate gate bias.
    """

    def __init__(self, input_size: tuple[int, int], num_hiddens: int,
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (
                num_cities, num_attrs).
            num_hiddens (int): The number of hidden units.
            device (torch.device): The device to put the parameters on.
        """
        super(LSTMCell, self).__init__(input_size=input_size, device=device)
        self.num_hiddens = num_hiddens
        for gate in ('i', 'f', 'o', 'g'):
            w, u, b = _gate_params(self.num_cities, self.num_attrs,
                                   self.num_hiddens, self.device)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(
        self, input: torch.Tensor,
        state: Optional[tuple[torch.Tensor,
                              torch.Tensor]]) -> tuple[torch.Tensor]:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.
            state (tuple[torch.Tensor, torch.Tensor]): The state of the cell.

        Returns:
            tuple[torch.Tensor]: The the new state of the cell.
        """
        x = input.reshape(-1, self.num_attrs)
        if state is None:
            h = torch.zeros(input.shape[0],
                            input.shape[1],
                            self.num_hiddens,
                            device=self.device)
            c = torch.zeros(input.shape[0],
                            input.shape[1],
                            self.num_hiddens,
                            device=self.device)
        else:
            h, c = state
        h = h.reshape(-1, self.num_hiddens)
        c = c.reshape(-1, self.num_hiddens)
        i = torch.sigmoid(
            torch.matmul(x, self.w_i) + torch.matmul(h, self.u_i) + self.b_i)
        f = torch.sigmoid(
            torch.matmul(x, self.w_f) + torch.matmul(h, self.u_f) + self.b_f)
        o = torch.sigmoid(
            torch.matmul(x, self.w_o) + torch.matmul(h, self.u_o) + self.b_o)
        g = torch.tanh(
            torch.matmul(x, self.w_g) + torch.matmul(h, self.u_g) + self.b_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h.reshape(-1, self.num_cities, self.num_hiddens),
                c.reshape(-1, self.num_cities, self.num_hiddens))


class GLSTMCell(LSTMCell):
    """A class for the GLSTM cell.

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to put the parameters on.
        w_i (Parameter): The input gate weight.
        u_i (Parameter): The input gate weight.
        b_i (Parameter): The input gate bias.
        w_f (Parameter): The forget gate weight.
        u_f (Parameter): The forget gate weight.
        b_f (Parameter): The forget gate bias.
        w_o (Parameter): The output gate weight.
        u_o (Parameter): The output gate weight.
        b_o (Parameter): The output gate bias.
        w_g (Parameter): The candidate gate weight.
        u_g (Parameter): The candidate gate weight.
        b_g (Parameter): The candidate gate bias.
        w_adj (Parameter): The adjacency matrix.
    """

    def __init__(self, input_size: tuple[int, int], num_hiddens: int,
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (
                num_cities, num_attrs).
            num_hiddens (int): The number of hidden units.
            device (torch.device): The device to put the parameters on.
        """
        super(GLSTMCell, self).__init__(input_size=input_size,
                                        num_hiddens=num_hiddens,
                                        device=device)
        self.w_adj = _get_param((self.num_cities, self.num_cities),
                                device=self.device)

    def forward(
        self, input: torch.Tensor,
        state: Optional[tuple[torch.Tensor,
                              torch.Tensor]]) -> tuple[torch.Tensor]:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.
            state (tuple[torch.Tensor, torch.Tensor]): The state of the cell.

        Returns:
            tuple[torch.Tensor]: The the new state of the cell.
        """
        x = input.reshape(-1, self.num_attrs)
        if state is None:
            h = torch.zeros(input.shape[0],
                            input.shape[1],
                            self.num_hiddens,
                            device=self.device)
            c = torch.zeros(input.shape[0],
                            input.shape[1],
                            self.num_hiddens,
                            device=self.device)
        else:
            h, c = state

        h = h.permute(0, 2, 1).reshape(-1, self.num_cities)
        h_adj = torch.matmul(h, self.w_adj)
        h = h.reshape(-1, self.num_hiddens, self.num_cities).permute(0, 2, 1)
        h_adj = h_adj.reshape(-1, self.num_hiddens,
                              self.num_cities).permute(0, 2, 1)

        h = h.reshape(-1, self.num_hiddens)
        c = c.reshape(-1, self.num_hiddens)
        i = torch.sigmoid(
            torch.matmul(x, self.w_i) + torch.matmul(h_adj, self.u_i) +
            self.b_i)
        f = torch.sigmoid(
            torch.matmul(x, self.w_f) + torch.matmul(h, self.u_f) + self.b_f)
        o = torch.sigmoid(
            torch.matmul(x, self.w_o) + torch.matmul(h_adj, self.u_o) +
            self.b_o)
        g = torch.tanh(
            torch.matmul(x, self.w_g) + torch.matmul(h_adj, self.u_g) +
            self.b_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h.reshape(-1, self.num_cities, self.num_hiddens),
                c.reshape(-1, self.num_cities, self.num_hiddens))


class GRUCell(Map):
    """A class for the GRU cell. It is a GRU cell with a map as input. The 
    input is a tensor of shape (batch_size, num_cities, num_attrs).

    Attributes:
        num_cities (int): The number of cities.
        num_attrs (int): The number of attributes.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to put the parameters on.
        w_z (Parameter): The update gate weight.
        u_z (Parameter): The update gate weight.
        b_z (Parameter): The update gate bias.
        w_r (Parameter): The reset gate weight.
        u_r (Parameter): The reset gate weight.
        b_r (Parameter): The reset gate bias.
        w_h (Parameter): The candidate gate weight.
        u_h (Parameter): The candidate gate weight.
        b_h (Parameter): The candidate gate bias.
    """

    def __init__(self, input_size: tuple[int, int], num_hiddens: int,
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_size (tuple[int, int]): The input size of shape (
                num_cities, num_attrs).
            num_hiddens (int): The number of hidden units.
            device (torch.device): The device to put the parameters on.
        """
        super(GRUCell, self).__init__(input_size=input_size, device=device)
        self.num_hiddens = num_hiddens
        for gate in ('z', 'r', 'h'):
            w, u, b = _gate_params(self.num_cities, self.num_attrs,
                                   self.num_hiddens, self.device)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(self, input: torch.Tensor,
                state: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.
            state (torch.Tensor): The state of the cell.

        Returns:
            torch.Tensor: The the new state of the cell.
        """
        x = input.reshape(-1, self.num_attrs)
        if state is None:
            h = torch.zeros(input.shape[0],
                            input.shape[1],
                            self.num_hiddens,
                            device=self.device)
        else:
            h = state
        h = h.reshape(-1, self.num_hiddens)
        z = torch.sigmoid(
            torch.matmul(x, self.w_z) + torch.matmul(h, self.u_z) + self.b_z)
        r = torch.sigmoid(
            torch.matmul(x, self.w_r) + torch.matmul(h, self.u_r) + self.b_r)
        h_tilde = torch.tanh(
            torch.matmul(x, self.w_h) + torch.matmul(r * h, self.u_h) +
            self.b_h)
        h = (1 - z) * h + z * h_tilde
        return h.reshape(-1, self.num_cities, self.num_hiddens)


class Dense(torch.nn.Module):
    """A class for the Dense layer. It is a dense layer with a map as input.
    The input is a tensor of shape (batch_size, num_cities, input_hiddens).

    Attributes:
        num_cities (int): The number of cities.
        input_hiddens (int): The number of hidden units of the input.
        num_hiddens (int): The number of hidden units.
        device (torch.device): The device to put the parameters on.
        w (Parameter): The weight.
        b (Parameter): The bias.
    """

    def __init__(self, input_shape: tuple[int, int], num_hiddens: int,
                 device: torch.device) -> None:
        """Initializes the class.

        Args:
            input_shape (tuple[int, int]): The input shape of shape (
                num_cities, input_hiddens).
            num_hiddens (int): The number of hidden units.
            device (torch.device): The device to put the parameters on.
        """
        super(Dense, self).__init__()
        self.num_cities, self.input_hiddens = input_shape
        self.num_hiddens = num_hiddens
        self.device = device
        self.w = _get_param((self.input_hiddens, self.num_hiddens),
                            device=self.device)
        self.b = _get_bias((self.num_cities, self.num_hiddens),
                           device=self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = input.reshape(-1, self.input_hiddens)
        return torch.matmul(x, self.w).reshape(-1, self.num_cities,
                                               self.num_hiddens) + self.b
