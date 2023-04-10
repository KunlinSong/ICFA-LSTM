from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

__all__ = [
    'Dense', 'GRUCell', 'ICA', 'ICFA', 'LSTMCell', 'MaxMinNorm', 'RNNCell'
]


def _get_assoc_mat(size: int, device: torch.device):
    mat = torch.rand(size, size, dtype=torch.float64, device=device) *2 - 1
    mat.fill_diagonal_(1)
    return Parameter(mat)

def _to_range(min_val: Union[int, float], max_val: Union[int, float], mat: torch.Tensor):
    mask = (mat < min_val) | (mat > max_val)
    mat[mask] = torch.rand(mask.sum()) * (max_val - min_val) + min_val
    return mat


def _get_param(shape: tuple[int], device: torch.device) -> Parameter:
    """Gets a parameter with the given shape and device.

    Args:
        shape (tuple[int]): The shape of the parameter.
        device (torch.device): The device of the parameter.

    Returns:
        Parameter: The parameter.
    """
    return Parameter(torch.randn(shape, dtype=torch.float64, device=device))


def _get_bias(shape: tuple[int], device: torch.device) -> Parameter:
    """Gets a bias with the given shape and device.

    Args:
        shape (tuple[int]): The shape of the bias.
        device (torch.device): The device of the bias.

    Returns:
        Parameter: The bias.
    """
    return Parameter(torch.zeros(shape, dtype=torch.float64, device=device))


def _gate_params(
        map_units: int, num_attrs: int, hidden_units: int,
        device: torch.device) -> tuple[Parameter, Parameter, Parameter]:
    """Gets the parameters for the gate.

    Args:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the parameters.

    Returns:
        tuple[Parameter, Parameter, Parameter]: The parameters for the gates.
    """
    return (_get_param((num_attrs, hidden_units), device),
            _get_param((hidden_units, hidden_units),
                       device), _get_bias((map_units, hidden_units), device))


class MaxMinNorm(nn.Module):
    """Max-min normalization.

    Attributes:
        max_vals (torch.Tensor): The maximum values.
    """

    def __init__(self) -> None:
        """Initializes the max-min normalization."""
        super(MaxMinNorm, self).__init__()
        self.max_vals = 0.0

    def forward(self,
                input: torch.Tensor,
                inverse: Optional[bool] = False) -> torch.Tensor:
        """Performs max-min normalization or inverse normalization.

        Args:
            input (torch.Tensor): The input that needs to be normalized or 
                inverse normalized.
            inverse (bool, optional): Whether to perform inverse normalization 
                or not. Defaults to False.

        Returns:
            torch.Tensor: The normalized or inverse normalized input.
        """
        if inverse:
            return input * self.max_vals
        max_vals = torch.abs(input)
        for dim in range(1, len(max_vals.shape)):
            self.max_vals = torch.max(max_vals, dim=dim, keepdim=True).values
        return input / self.max_vals


class Map(nn.Module):
    """The base class for the map.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device) -> None:
        """Initializes the map.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(Map, self).__init__()
        self.map_units = map_units
        self.num_attrs = num_attrs
        self.hidden_units = hidden_units
        self.device = device


class ICA(Map):
    """The ICA layer.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        w_assoc (torch.Tensor): The association weights.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device) -> None:
        """Initializes the ICA map.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(ICA, self).__init__(map_units, num_attrs, hidden_units, device)
        self.w_assoc = _get_assoc_mat(map_units, device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        self.w_assoc = _to_range(-1, 1, self.w_assoc)
        x = torch.transpose(input, -1, -2)
        y = torch.reshape(y, (-1, self.map_units))
        y = torch.matmul(y, self.w_assoc)
        y = y.reshape(x.shape)
        return torch.transpose(y, -1, -2)


class ICFA(Map):
    """The ICFA layer.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        w_assoc_[i] (torch.Tensor): The association weights for the i-th 
            attribute.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device) -> None:
        """Initializes the ICFA map.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(ICFA, self).__init__(map_units, num_attrs, hidden_units, device)
        for idx in range(num_attrs):
            w_assoc = _get_param((map_units, map_units), device)
            w_assoc.fill_diagonal_(1)
            setattr(self, f'w_assoc_{idx}', w_assoc)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        attrs_inputs = torch.split(input, 1, dim=-1)
        result = []
        for idx, mat in enumerate(attrs_inputs):
            w_assoc = getattr(self, f'w_assoc_{idx}')
            w_assoc = _to_range(-1, 1, w_assoc)
            w_assoc.fill_diagonal_(1)
            x = torch.squeeze(mat, -1)
            y = torch.matmul(x, w_assoc)
            y = torch.unsqueeze(y, -1)
            result.append(y)
        return torch.cat(result, -1)


class LSTMCell(Map):
    """The LSTM cell.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        w_i (torch.Tensor): The input gate weights.
        u_i (torch.Tensor): The input gate recurrent weights.
        b_i (torch.Tensor): The input gate bias.
        w_f (torch.Tensor): The forget gate weights.
        u_f (torch.Tensor): The forget gate recurrent weights.
        b_f (torch.Tensor): The forget gate bias.
        w_o (torch.Tensor): The output gate weights.
        u_o (torch.Tensor): The output gate recurrent weights.
        b_o (torch.Tensor): The output gate bias.
        w_g (torch.Tensor): The gate weights.
        u_g (torch.Tensor): The gate recurrent weights.
        b_g (torch.Tensor): The gate bias.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device) -> None:
        """Initializes the LSTM cell.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(LSTMCell, self).__init__(map_units, num_attrs, hidden_units,
                                       device)
        for gate in ('i', 'f', 'o', 'g'):
            w, u, b = _gate_params(map_units, num_attrs, hidden_units, device)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.
            state (tuple[torch.Tensor, torch.Tensor], optional): The state 
                tensor. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The new state of the cell.
        """
        if state is None:
            h = torch.zeros(
                (input.shape[0], self.map_units, self.hidden_units),
                dtype=torch.float64,
                device=self.device)
            c = torch.zeros(
                (input.shape[0], self.map_units, self.hidden_units),
                dtype=torch.float64,
                device=self.device)
        else:
            h, c = state
        output_shape = h.shape
        x = torch.reshape(input, (-1, self.num_attrs))
        h = torch.reshape(h, (-1, self.hidden_units))
        i = torch.sigmoid(
            torch.reshape(
                torch.matmul(x, self.w_i) +
                torch.matmul(h, self.u_i), output_shape) + self.b_i)
        f = torch.sigmoid(
            torch.reshape(
                torch.matmul(x, self.w_f) +
                torch.matmul(h, self.u_f), output_shape) + self.b_f)
        o = torch.sigmoid(
            torch.reshape(
                torch.matmul(x, self.w_o) +
                torch.matmul(h, self.u_o), output_shape) + self.b_o)
        g = torch.tanh(
            torch.reshape(
                torch.matmul(x, self.w_g) +
                torch.matmul(h, self.u_g), output_shape) + self.b_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h, c)


class GRUCell(Map):
    """The GRU cell.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        w_z (torch.Tensor): The update gate weights.
        u_z (torch.Tensor): The update gate recurrent weights.
        b_z (torch.Tensor): The update gate bias.
        w_r (torch.Tensor): The reset gate weights.
        u_r (torch.Tensor): The reset gate recurrent weights.
        b_r (torch.Tensor): The reset gate bias.
        w_h (torch.Tensor): The hidden gate weights.
        u_h (torch.Tensor): The hidden gate recurrent weights.
        b_h (torch.Tensor): The hidden gate bias.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device) -> None:
        """Initializes the GRU cell.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(GRUCell, self).__init__(map_units, num_attrs, hidden_units,
                                      device)
        for gate in ('z', 'r', 'h'):
            w, u, b = _gate_params(map_units, num_attrs, hidden_units, device)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[tuple[torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.
            state (tuple[torch.Tensor], optional): The state tensor. 
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The new state of the cell.
        """
        if state is None:
            h = torch.zeros(
                (input.shape[0], self.map_units, self.hidden_units),
                dtype=torch.float64,
                device=self.device)
        else:
            h = state[0]
        output_shape = h.shape
        x = torch.reshape(input, (-1, self.num_attrs))
        h = torch.reshape(h, (-1, self.hidden_units))
        z = torch.sigmoid(
            torch.reshape(
                torch.matmul(x, self.w_z) +
                torch.matmul(h, self.u_z), output_shape) + self.b_z)
        r = torch.sigmoid(
            torch.reshape(
                torch.matmul(x, self.w_r) +
                torch.matmul(h, self.u_r), output_shape) + self.b_r)
        h_ = torch.tanh(
            torch.reshape(
                torch.matmul(x, self.w_h) +
                torch.matmul(r * h, self.u_h), output_shape) + self.b_h)
        h = (1 - z) * h + z * h_
        return (h,)


class RNNCell(Map):
    """The RNN cell.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        dense_i (Dense): The dense layer for the input.
        dense_h (Dense): The dense layer for the hidden state.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device):
        """Initializes the RNN cell.

        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super(RNNCell, self).__init__(map_units, num_attrs, hidden_units,
                                      device)
        self.dense_i = Dense(map_units, num_attrs, hidden_units, device)
        self.dense_h = Dense(map_units, hidden_units, hidden_units, device)

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[tuple[torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.
            state (tuple[torch.Tensor], optional): The state tensor. Defaults 
                to None.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The new state of the cell.
        """
        if state is None:
            h = torch.zeros(
                (input.shape[0], self.map_units, self.hidden_units),
                dtype=torch.float64,
                device=self.device)
        else:
            h = state[0]
        h = torch.tanh(self.dense_i(input) + self.dense_h(h))
        return (h,)


class Dense(Map):
    """The dense layer.

    Attributes:
        map_units (int): The number of map units.
        num_attrs (int): The number of attributes.
        hidden_units (int): The number of hidden units.
        device (torch.device): The device of the map.
        w (torch.Tensor): The weights.
        b (torch.Tensor): The bias.
    """

    def __init__(self, map_units: int, num_attrs: int, hidden_units: int,
                 device: torch.device):
        """Initializes the dense layer.
        
        Args:
            map_units (int): The number of map units.
            num_attrs (int): The number of attributes.
            hidden_units (int): The number of hidden units.
            device (torch.device): The device of the map layer.
        """
        super().__init__(map_units, num_attrs, hidden_units, device)
        self.w = _get_param((num_attrs, hidden_units), device)
        self.b = _get_bias((map_units, hidden_units), device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        output_shape = list(input.shape)
        output_shape[-1] = self.hidden_units
        x = torch.reshape(input, (-1, self.num_attrs))
        return torch.reshape(torch.matmul(x, self.w), output_shape) + self.b
