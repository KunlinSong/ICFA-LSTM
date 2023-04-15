import torch

from icfalstm.types import *


class ArcTanNorm(torch.nn.Module):
    """A normalization layer that normalizes the input to the range of 
    [-1, 1] or [0, 1] using the arctan function.
    
    .. math::

        \begin{array}{ll}
        y = \frac{2\arctan(x)}{\pi} \quad if \quad keep \_ zero \_ position \\
        y = \frac{\arctan(x)}{\pi} + 0.5 \quad if \quad not \quad keep \_ zero 
            \_ position \\
        \end{array}
    
    Inputs: x
        - **x** (Tensor): A tensor of shape `(batch_size, *)`
        - **inverse** (bool): Whether to inverse the normalization. Defaults 
            to False.
    
    Outputs: y
        - **y** (Tensor): A tensor of shape `(batch_size, *)`
    
    Attributes:
        keep_zero_position (bool): Whether to keep the zero position of the 
            input. If True, the output will be in the range of [-1, 1]. If 
            False, the output will be in the range of [0, 1].
    """
    torch.nn.LayerNorm
    def __init__(self, keep_zero_position: bool = True) -> None:
        """initializes an arctan normalization layer.
        
        Args:
            keep_zero_position: Whether to keep the zero position of the 
                input. If True, the output will be in the range of [-1, 1]. 
                If False, the output will be in the range of [0, 1]. 
                Defaults to True.
        """
        super().__init__()
        self.keep_zero_position = keep_zero_position
    
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            inverse_norm = x / 2 if self.keep_zero_position else x - 0.5
            return torch.tan(inverse_norm * torch.pi)
        else:
            norm = torch.atan(x) / torch.pi
            return norm * 2 if self.keep_zero_position else norm + 0.5

class MaxMinNorm(torch.nn.Module):
    """A normalization layer that normalizes the input to the range of 
    [-1, 1] or [0, 1] using the max-min normalization.

    .. math::

        \begin{array}{ll}
        y = \frac{x}{\max(\abs(x))} \quad if \quad keep \_ zero \_ position \\
        y = \frac{x - \min(x)}{\max(x) - \min(x)} \quad if \quad not \quad 
            keep \_ zero \_ position \\
        \end{array}
    
    Inputs: x
        - **x** (Tensor): A tensor of shape `(batch_size, *)`
        - **inverse** (bool): Whether to inverse the normalization. Defaults 
            to False.
    
    Outputs: y
        - **y** (Tensor): A tensor of shape `(batch_size, *)`
    
    Attributes:
        keep_zero_position (bool): Whether to keep the zero position of the 
            input. If True, the output will be in the range of [-1, 1]. If 
            False, the output will be in the range of [0, 1].
    """
    def __init__(self, keep_zero_position: bool = True) -> None:
        """initializes a max-min normalization layer.
        
        Args:
            keep_zero_position: Whether to keep the zero position of the 
                input. If True, the output will be in the range of [-1, 1].
                If False, the output will be in the range of [0, 1]. Defaults 
                to True.
        """
        super().__init__()
        self.keep_zero_position = keep_zero_position
        self.min_vals = 0.0
        self.max_vals = 0.0
    
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return x * (self.max_vals - self.min_vals) + self.min_vals
        else:
            max_vals = x
            min_vals = x
            for dim in range(1, x.dim()):
                max_vals = torch.max(max_vals, dim=dim, keepdim=True).values
                min_vals = torch.min(min_vals, dim=dim, keepdim=True).values
            if self.keep_zero_position:
                self.max_vals = torch.max(max_vals, torch.abs(min_vals))
                self.min_vals = 0.0
            else:
                self.max_vals = max_vals
                self.min_vals = min_vals
            return (x - self.min_vals) / (self.max_vals - self.min_vals)
            
        