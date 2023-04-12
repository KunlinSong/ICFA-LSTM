import torch

from icfalstm.types import *


def get_weight_param(*size: int,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None) -> Parameter:
    """Gets a weight parameter of the given size.

    Args:
        size: The size of the weight parameter.
        dtype: The data type of the weight parameter. Defaults to None.
        device: The device of the weight parameter. Defaults to None.
    
    Returns:
        A weight parameter of the given size from a normal distribution.
    """

    mat = torch.randn(*size, dtype=dtype, device=device)
    return torch.nn.Parameter(mat)


def get_bias_param(*size: int,
                   dtype: Optional[torch.dtype] = None,
                   device: Optional[torch.device] = None) -> Parameter:
    """Gets a bias parameter of the given size.

    Args:
        size: The size of the bias parameter.
        dtype: The data type of the bias parameter. Defaults to None.
        device: The device of the bias parameter. Defaults to None.
    
    Returns:
        A bias parameter of the given size of zeros.
    """
    mat = torch.zeros(*size, dtype=dtype, device=device)
    return torch.nn.Parameter(mat)


torch.nn.Linear


class Dense(torch.nn.Module):
    """A dense layer, which is a linear layer with a bias, that can calculate 
    the output of map units.

    .. math::

        \begin{array}{ll}
        y = W x + b \\
        \end{array}
    
    Inputs: x
        - **x** (Tensor): A tensor of shape '(batch_size, map_units, 
            in_features)'. or '(map_units, in_features)'. Tensor containing 
            the input features of the map units.
    
    Outputs: y
        - **y** (Tensor): A tensor of shape '(batch_size, map_units, 
            out_features)' or '(map_units, out_features)'. Tensor containing 
            the output features of the map units.

    Attributes:
        map_units: The number of map units.
        in_features: The number of input features.
        out_features: The number of output features.
        bias: Whether to use a bias. Defaults to True.
        dtype: The data type of the dense layer.
        device: The device of the dense layer.
        weight: The weight parameter of the dense layer.
        bias: The bias parameter of the dense layer.
    """

    def __init__(
        self,
        map_units: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """initializes a dense layer.
        
        Args:
            map_units: The number of map units.
            in_features: The number of input features.
            out_features: The number of output features.
            bias: Whether to use a bias. Defaults to True.
            dtype: The data type of the dense layer. Defaults to None.
            device: The device of the dense layer. Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.in_features = in_features
        self.out_features = out_features
        self.param_kwargs = {'dtype': dtype, 'device': device}
        self.weight = get_weight_param(self.out_features, self.in_features,
                                       **self.param_kwargs)
        self.bias = (get_bias_param(self.map_units, self.out_features, **
                                    self.param_kwargs) if bias else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            (x.dim() in (2, 3)) and
            (x.shape[-2:] == (self.map_units, self.in_features))
        ), f"""Dense: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), in_features({self.in_features})) or 2-D of 
            shape (map_units({self.map_units}), in_features({self.in_features})
            ), but received {x.dim()}-D tensor of shape {x.shape}"""

        is_batched = x.dim() == 3
        if not is_batched:
            x = torch.unsqueeze(x, 0)
        x = torch.transpose(x, 0, 2)
        to_shape = (self.out_features,) + x.shape[1:]
        x = torch.reshape(x, (self.in_features, -1))
        y = torch.matmul(self.weight, x)
        y = torch.reshape(y, to_shape)
        y = torch.transpose(y, 0, 2)
        if not is_batched:
            y = torch.squeeze(y, 0)
        return y if self.bias is None else y + self.bias
