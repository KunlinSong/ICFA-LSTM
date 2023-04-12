import torch

import icfalstm.nn.layer.basic as basic
from icfalstm.types import *


def _get_gate_params(
    map_units: int,
    input_size: int,
    hidden_size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> Tuple[Parameter, Parameter, Parameter]:
    """Gets the parameters for the gates of an LSTM or GRU cell.
    
    Args:
        map_units: The number of map units.
        input_size: The number of features in the input.
        hidden_size: The number of features in the hidden state.
        dtype: The data type of the parameters. Defaults to None.
        device: The device of the parameters. Defaults to None.
    """
    param_kwargs = {'dtype': dtype, 'device': device}
    return (basic.get_weight_param(hidden_size, input_size, **param_kwargs),
            basic.get_weight_param(hidden_size, hidden_size, **param_kwargs),
            basic.get_bias_param(map_units, hidden_size, **param_kwargs))


class LSTMCell(torch.nn.Module):
    """A long short-term memory (LSTM) cell.
    
    The LSTM cell is different from torch.nn.LSTMCell in that it can calculate 
    the output of multiple map units at once.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{i} x + W_{i} h_{t-1} + b_{i}) \\
        f = \sigma(W_{f} x + W_{f} h_{t-1} + b_{f}) \\
        o = \sigma(W_{o} x + W_{o} h_{t-1} + b_{o}) \\
        g = \tanh(W_{g} x + W_{g} h_{t-1} + b_{g}) \\
        c_{t} = f * c_{t-1} + i * g \\
        h_{t} = o * \tanh(c_{t}) \\
        \end{array}
    
    where :math:`\sigma` is the sigmoid function, :math:`\tanh` is the tanh 
        function, and :math:`*` is the Hadamard product.

    Inputs: x, (h_{t-1}, c_{t-1})
        - **x** (Tensor): A tensor of shape `(batch_size, map_units, 
            input_size)` or `(map_units, input_size)`. Tensor containing the 
            input features of the map units.
        - **h_{t-1}** (Tensor): A tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing 
            the hidden state for the map units from the previous time step.
        - **c_{t-1}** (Tensor): A tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing 
            the cell state for the map units from the previous time step.

        If `(h_{t-1}, c_{t-1})` is not provided, both **h_0** and **c_0** 
        default to zeros.
    
    Outputs: (h_{t}, c_{t})
        - **h_{t}** (Tensor): A tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing 
            the hidden state for the map units from the current time step.
        - **c_{t}** (Tensor): A tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing 
            the cell state for the map units from the current time step.
    
    Attributes:
        map_units (int): The number of map units.
        input_size (int): The number of features in the input :math:`x`.
        hidden_size (int): The number of features in the hidden state :math:`h`.
        param_kwargs (Dict[str, Any]): The keyword arguments for creating 
            parameters.
        w_i (Parameter): The learnable input gate weights of shape 
            `(hidden_size, input_size)`.
        w_f (Parameter): The learnable forget gate weights of shape 
            `(hidden_size, input_size)`.
        w_o (Parameter): The learnable output gate weights of shape 
            `(hidden_size, input_size)`.
        w_g (Parameter): The learnable cell gate weights of shape 
            `(hidden_size, input_size)`.
        u_i (Parameter): The learnable input gate weights of shape 
            `(hidden_size, hidden_size)`.
        u_f (Parameter): The learnable forget gate weights of shape 
            `(hidden_size, hidden_size)`.
        u_o (Parameter): The learnable output gate weights of shape 
            `(hidden_size, hidden_size)`.
        u_g (Parameter): The learnable cell gate weights of shape 
            `(hidden_size, hidden_size)`.
        b_i (Parameter): The learnable input gate bias of shape 
            `(map_units, hidden_size)`.
        b_f (Parameter): The learnable forget gate bias of shape 
            `(map_units, hidden_size)`.
        b_o (Parameter): The learnable output gate bias of shape 
            `(map_units, hidden_size)`.
        b_g (Parameter): The learnable cell gate bias of shape 
            `(map_units, hidden_size)`.
        dim_changer (basic.DimensionChanger): The dimension changer.
    """

    def __init__(self,
                 map_units: int,
                 input_size: int,
                 hidden_size: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        """Initialize the LSTM cell.
        
        Args:
            map_units (int): The number of map units.
            input_size (int): The number of features in the input :math:`x`.
            hidden_size (int): The number of features in the hidden state 
                :math:`h`.
            dtype (torch.dtype, optional): The data type of the parameters.
                Defaults to None.
            device (torch.device, optional): The device of the parameters.
                Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param_kwargs = {'dtype': dtype, 'device': device}
        for gate in ('i', 'f', 'o', 'g'):
            w, u, b = _get_gate_params(map_units, input_size, hidden_size,
                                      **self.param_kwargs)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)
        self.dim_changer = basic.DimensionChanger(map_units)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            (x.dim() in (2, 3)) and
            (x.shape[-2:] == (self.map_units, self.input_size))
        ), f"""LSTMCell: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), input_size({self.input_size})) or 2-D of 
            shape (map_units({self.map_units}), input_size({self.input_size}
            )), but recieved {x.dim()}-D tensor of shape {x.shape}"""
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0], self.map_units, self.hidden_size,
                            **self.param_kwargs)
            c = torch.zeros(x.shape[0], self.map_units, self.hidden_size,
                            **self.param_kwargs)
        else:
            h = torch.squeeze(state[0], 0) if is_batched else state[0]
            c = torch.squeeze(state[1], 0) if is_batched else state[1]
        x = self.dim_changer.three_to_two(x)
        h = self.dim_changer.three_to_two(h)
        i = torch.sigmoid(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_i, x) + torch.matmul(self.u_i, h)) +
            self.b_i)
        f = torch.sigmoid(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_f, x) + torch.matmul(self.u_f, h)) +
            self.b_f)
        o = torch.sigmoid(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_o, x) + torch.matmul(self.u_o, h)) +
            self.b_o)
        g = torch.tanh(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_g, x) + torch.matmul(self.u_g, h)) +
            self.b_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return (h, c) if is_batched else (h.squeeze(0), c.squeeze(0))


torch.nn.GRUCell


class GRUCell(torch.nn.Module):
    """A gated recurrent unit (GRU) cell.
    
    The GRU cell is different from torch.nn.GRUCell in that it can calculate 
    the output for multiple map units at once.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{r} x + U_{r} h_{t-1} + B_{r}) \\
        z = \sigma(W_{z} x + U_{z} h_{t-1} + B_{z}) \\
        n = \tanh(W_{n} x + U_{n} (r * h_{t-1}) + b_{n})) \\
        h_{t} = (1 - z) * n + z * h_{t-1}
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard 
        product.
    
    Inputs: x, (h_{t-1},)
        - **x** (torch.Tensor): tensor of shape `(batch_size, map_units, 
            input_size)` or `(map_units, input_size)`. Tensor containing the 
            input features of the map units.
        - **h_{t-1}** (torch.Tensor): tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing the 
            hidden state of the map units from the previous time step.
        
        If `(h_{t-1},)` is not provided, **h_{t-1}** will default to zeros.
    
    Outputs: (h_{t},)
        - **h_{t}** (torch.Tensor): tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing the 
            hidden state of the map units from the current time step.
    
    Attributes:
        map_units (int): The number of map units.
        input_size (int): The number of expected features in the input 
            :math:`x`.
        hidden_size (int): The number of features in the hidden state :math:`h`.
        param_kwargs (dict): The keyword arguments for the parameters.
        w_r (torch.Tensor): The learnable weights of the GRU cell for the 
            reset gate, of shape `(map_units, hidden_size, input_size)`.
        u_r (torch.Tensor): The learnable weights of the GRU cell for the 
            reset gate, of shape `(map_units, hidden_size, hidden_size)`.
        b_r (torch.Tensor): The learnable bias of the GRU cell for the reset 
            gate, of shape `(map_units, hidden_size)`.
        w_z (torch.Tensor): The learnable weights of the GRU cell for the 
            update gate, of shape `(map_units, hidden_size, input_size)`.
        u_z (torch.Tensor): The learnable weights of the GRU cell for the 
            update gate, of shape `(map_units, hidden_size, hidden_size)`.
        b_z (torch.Tensor): The learnable bias of the GRU cell for the update 
            gate, of shape `(map_units, hidden_size)`.
        w_n (torch.Tensor): The learnable weights of the GRU cell for the 
            new gate, of shape `(map_units, hidden_size, input_size)`.
        u_n (torch.Tensor): The learnable weights of the GRU cell for the 
            new gate, of shape `(map_units, hidden_size, hidden_size)`.
        b_n (torch.Tensor): The learnable bias of the GRU cell for the new 
            gate, of shape `(map_units, hidden_size)`.
        dim_changer (basic.DimensionChanger): The dimension changer.
    """

    def __init__(self,
                 map_units: int,
                 input_size: int,
                 hidden_size: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        """Initialize the GRU cell.
        
        Args:
            map_units (int): The number of map units.
            input_size (int): The number of expected features in the input 
                :math:`x`.
            hidden_size (int): The number of features in the hidden state 
                :math:`h`.
            dtype (torch.dtype, optional): The data type of the parameters.
                Defaults to None.
            device (torch.device, optional): The device of the parameters.
                Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param_kwargs = {'dtype': dtype, 'device': device}
        for gate in ('r', 'z', 'n'):
            w, u, b = _get_gate_params(map_units, input_size, hidden_size,
                                      **self.param_kwargs)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)
        self.dim_changer = basic.DimensionChanger(map_units)

    def forward(self,
                x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor]] = None
               ) -> Tuple[torch.Tensor]:
        assert (
            (x.dim() in (2, 3)) and
            (x.shape[-2:] == (self.map_units, self.input_size))
        ), f"""GRUCell: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), input_size({self.input_size})) or 2-D of shape 
            (map_units({self.map_units}), input_size({self.input_size})), but 
            recieved {x.dim()}-D tensor of shape {x.shape}"""
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0], self.map_units, self.hidden_size,
                            **self.param_kwargs)
        else:
            h = torch.squeeze(state[0], 0) if is_batched else state[0]
        x = self.dim_changer.three_to_two(x)
        h = self.dim_changer.three_to_two(h)
        r = torch.sigmoid(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_r, x) + torch.matmul(self.u_r, h)) +
            self.b_r)
        z = torch.sigmoid(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_z, x) + torch.matmul(self.u_z, h)) +
            self.b_z)
        n = torch.tanh(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_n, x) +
                torch.matmul(self.u_n,
                             self.dim_changer.three_to_two(r) * h)) + self.b_n)
        h = (1 - z) * n + z * h
        return (h,) if is_batched else (h.squeeze(0),)


class RNNCell(torch.nn.Module):
    """A simple RNN cell.
    
    The RNN cell is different from torch.nn.RNNCell in that it can calculate 
    the output of multiple map units at once.

    .. math::

        \begin{array}{ll}
        h_{t} = \tanh(W_{i} x + b_{i}  +  W_{h} h_{t-1} + b_{h})
        \end{array}
    
    Inputs: x, (h_{t-1})
        - **x** (torch.Tensor): Tensor of shape `(batch_size, map_units, 
            input_size)` containing the features of the map units.
        - **h_{t-1}** (torch.Tensor, optional): Tensor of shape `(batch_size, 
            map_units, hidden_size)` containing the previous hidden state of 
            the map units. 

        If `(h_{t-1},)` is not provided, **h_{t-1}** will default to zeros.
    
    Outputs: h_{t}
        - **h_{t}** (torch.Tensor): Tensor of shape `(batch_size, map_units, 
            hidden_size)` containing the next hidden state of the map units.
    
    Attributes:
        map_units (int): The number of map units.
        input_size (int): The number of expected features in the input
            :math:`x`.
        hidden_size (int): The number of features in the hidden state 
            :math:`h`.
        param_kwargs (dict): The keyword arguments for the parameters.
        w_i (torch.Tensor): The learnable weight of the RNN cell for the input 
            :math:`x`, of shape `(hidden_size, input_size)`.
        w_h (torch.Tensor): The learnable weight of the RNN cell for the 
            previous hidden state :math:`h_{t-1}`, of shape
            `(hidden_size, hidden_size)`.
        b_i (torch.Tensor): The learnable bias of the RNN cell for the input 
            :math:`x`, of shape `(map_units, hidden_size)`.
        b_h (torch.Tensor): The learnable bias of the RNN cell for the 
            previous hidden state :math:`h_{t-1}`, of shape
            `(map_units, hidden_size)`.
        dim_changer (DimensionChanger): The dimension changer for the RNN cell.    
    """

    def __init__(self,
                 map_units: int,
                 input_size: int,
                 hidden_size: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        """Initializes the RNN cell.

        Args:
            map_units (int): The number of map units.
            input_size (int): The number of expected features in the input
                :math:`x`.
            hidden_size (int): The number of features in the hidden state 
                :math:`h`.
            dtype (torch.dtype, optional): The desired data type of the 
                parameters. Defaults to None.
            device (torch.device, optional): The desired device of the 
                parameters. Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param_kwargs = {'dtype': dtype, 'device': device}
        self.w_i = basic.get_weight_param(hidden_size, input_size,
                                          **self.param_kwargs)
        self.w_h = basic.get_weight_param(hidden_size, hidden_size,
                                          **self.param_kwargs)
        self.b_i = basic.get_bias_param(map_units, hidden_size,
                                        **self.param_kwargs)
        self.b_h = basic.get_bias_param(map_units, hidden_size,
                                        **self.param_kwargs)
        self.dim_changer = basic.DimensionChanger(map_units)

    def forward(self,
                x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor]] = None
               ) -> Tuple[torch.Tensor]:
        assert (
            (x.dim() in (2, 3)) and
            (x.shape[-2:] == (self.map_units, self.input_size))
        ), f"""RNNCell: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), input_size({self.input_size})) or 2-D of shape 
            (map_units({self.map_units}), input_size({self.input_size})), but 
            recieved {x.dim()}-D tensor of shape {x.shape}"""
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
        if state is None:
            h = torch.zeros(x.shape[0], self.map_units, self.hidden_size,
                            **self.param_kwargs)
        else:
            h = torch.squeeze(state[0], 0) if is_batched else state[0]
        x = self.dim_changer.three_to_two(x)
        h = self.dim_changer.three_to_two(h)

        h = torch.tanh(
            self.dim_changer.two_to_three(
                torch.matmul(self.w_i, x) + torch.matmul(self.w_h, h)) +
            self.b_i + self.b_h)
        return (h,) if is_batched else (h.squeeze(0),)