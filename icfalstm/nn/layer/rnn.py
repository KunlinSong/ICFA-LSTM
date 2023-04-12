import torch

import icfalstm.nn.layer.basic as basic
from icfalstm.types import *


def get_gate_params(
    map_units: int,
    input_size: int,
    hidden_size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> Tuple[Parameter, Parameter, Parameter]:
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
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    
    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard 
        product.

    Inputs: x, (h_{t-1}, c_{t-1})
        - **x** (Tensor): A tensor of shape '(batch_size, map_units, 
            input_size)'. or '(map_units, input_size)'. Tensor containing the 
            input features of the map units.
        - **h_{t-1}** (Tensor): A tensor of shape '(batch_size, map_units, 
            hidden_size)'. or '(map_units, hidden_size)'. Tensor containing 
            the hidden state for the map units from the previous time step.
        - **c_{t-1}** (Tensor): A tensor of shape '(batch_size, map_units, 
            hidden_size)'. or '(map_units, hidden_size)'. Tensor containing 
            the cell state for the map units from the previous time step.

        If `(h_{t-1}, c_{t-1})` is not provided, both **h_0** and **c_0** 
        default to zero.
    
    Outputs: (h_{t}, c_{t})
        - **h_{t}** (Tensor): A tensor of shape '(batch_size, map_units, 
            hidden_size)'. or '(map_units, hidden_size)'. Tensor containing 
            the hidden state for the map units from the current time step.
        - **c_{t}** (Tensor): A tensor of shape '(batch_size, map_units, 
            hidden_size)'. or '(map_units, hidden_size)'. Tensor containing 
            the cell state for the map units from the current time step.
    
    Attributes:
        map_units (int): The number of map units.
        input_size (int): The number of features in the input: math:`x`.
        hidden_size (int): The number of features in the hidden state: math:`h`.
        param_kwargs (Dict[str, Any]): The keyword arguments for creating 
            parameters.
        w_i (Parameter): The learnable input gate weights of shape 
            '(hidden_size, input_size)'.
        w_f (Parameter): The learnable forget gate weights of shape 
            '(hidden_size, input_size)'.
        w_o (Parameter): The learnable output gate weights of shape 
            '(hidden_size, input_size)'.
        w_g (Parameter): The learnable cell gate weights of shape 
            '(hidden_size, input_size)'.
        u_i (Parameter): The learnable input gate weights of shape 
            '(hidden_size, hidden_size)'.
        u_f (Parameter): The learnable forget gate weights of shape 
            '(hidden_size, hidden_size)'.
        u_o (Parameter): The learnable output gate weights of shape 
            '(hidden_size, hidden_size)'.
        u_g (Parameter): The learnable cell gate weights of shape 
            '(hidden_size, hidden_size)'.
        b_i (Parameter): The learnable input gate bias of shape 
            '(map_units, hidden_size)'.
        b_f (Parameter): The learnable forget gate bias of shape 
            '(map_units, hidden_size)'.
        b_o (Parameter): The learnable output gate bias of shape 
            '(map_units, hidden_size)'.
        b_g (Parameter): The learnable cell gate bias of shape 
            '(map_units, hidden_size)'.
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
            input_size (int): The number of features in the input: math:`x`.
            hidden_size (int): The number of features in the hidden state:
                math:`h`.
            dtype (torch.dtype, optional): The data type of the parameters.
                Default: None.
            device (torch.device, optional): The device of the parameters.
                Default: None.
        """
        super().__init__()
        self.map_units = map_units
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param_kwargs = {'dtype': dtype, 'device': device}
        for gate in ('i', 'f', 'o', 'g'):
            w, u, b = get_gate_params(map_units, input_size, hidden_size,
                                      **self.param_kwargs)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

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
        to_shape = (self.hidden_size, self.map_units, x.shape[0])
        x = torch.transpose(x, 0, 2)
        x = torch.reshape(x, (self.input_size, -1))
        h = torch.transpose(h, 0, 2)
        h = torch.reshape(h, (self.hidden_size, -1))
        i = torch.sigmoid(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_i, x) +
                    torch.matmul(self.u_i, h), to_shape), 0, 2) + self.b_i)
        f = torch.sigmoid(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_f, x) +
                    torch.matmul(self.u_f, h), to_shape), 0, 2) + self.b_f)
        o = torch.sigmoid(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_o, x) +
                    torch.matmul(self.u_o, h), to_shape), 0, 2) + self.b_o)
        g = torch.tanh(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_g, x) +
                    torch.matmul(self.u_g, h), to_shape), 0, 2) + self.b_g)
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
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
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
        
        If `(h_{t-1},)` is not provided, both **h_{t-1}** will default to zero.
    
    Outputs: (h_{t},)
        - **h_{t}** (torch.Tensor): tensor of shape `(batch_size, map_units, 
            hidden_size)` or `(map_units, hidden_size)`. Tensor containing the 
            hidden state of the map units from the current time step.
    
    Attributes:
        map_units (int): The number of map units.
        input_size (int): The number of expected features in the input:
            math:`x`.
        hidden_size (int): The number of features in the hidden state:
            math:`h`.
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
            input_size (int): The number of expected features in the input: 
                math:`x`.
            hidden_size (int): The number of features in the hidden state:
                math:`h`.
            dtype (torch.dtype, optional): The data type of the parameters.
                Default: None.
            device (torch.device, optional): The device of the parameters.
                Default: None.
        """
        super().__init__()
        self.map_units = map_units
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param_kwargs = {'dtype': dtype, 'device': device}
        for gate in ('r', 'z', 'n'):
            w, u, b = get_gate_params(map_units, input_size, hidden_size,
                                      **self.param_kwargs)
            setattr(self, f'w_{gate}', w)
            setattr(self, f'u_{gate}', u)
            setattr(self, f'b_{gate}', b)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (x.dim() in (2, 3)) and (
            x.shape[-2:] == (self.map_units, self.input_size)
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
        to_shape = (self.hidden_size, self.map_units, x.shape[0])
        x = torch.transpose(x, 0, 2)
        x = torch.reshape(x, (self.input_size, -1))
        h = torch.transpose(h, 0, 2)
        h = torch.reshape(h, (self.hidden_size, -1))
        r = torch.sigmoid(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_r, x) +
                    torch.matmul(self.u_r, h), to_shape), 0, 2) + self.b_r)
        z = torch.sigmoid(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_z, x) +
                    torch.matmul(self.u_z, h), to_shape), 0, 2) + self.b_z)
        n = torch.tanh(
            torch.transpose(
                torch.reshape(
                    torch.matmul(self.w_n, x) +
                    torch.matmul(self.u_n, r * h), to_shape), 0, 2) + self.b_n)
        h = (1 - z) * n + z * h
        return (h, ) if is_batched else (h.squeeze(0), )
