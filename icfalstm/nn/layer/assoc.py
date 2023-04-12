import torch

from icfalstm.nn.layer.basic import DimensionChanger
from icfalstm.types import *


class AssociationMatrix:
    """An association matrix.

    The association matrix is used to explain the inter-city association 
    between cities. The values are constrained to have values within the 
    range [min_val, max_val] and the diagonal is set to self_assoc_weight.
    The diagonal is set to self_assoc_weight, means the association of a 
    city to itself.

    Attributes:
        map_units: The number of map units.
        param_kwargs: The keyword arguments for the parameters.
        min_val: The minimum value of the association matrix.
        max_val: The maximum value of the association matrix.
        self_assoc_weight: The weight of the self association.
        mat: The association matrix, of shape (map_units, map_units).
    """

    def __init__(self,
                 map_units: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 min_val: int = -1,
                 max_val: int = 1,
                 self_assoc_weight: int = 1) -> None:
        """Initializes an association matrix.

        Args:
            map_units: The number of map units.
            dtype: The data type of the map layer. Defaults to None.
            device: The device of the map layer. Defaults to None.
            min_val: The minimum value of the association matrix. 
                Defaults to -1.
            max_val: The maximum value of the association matrix.
                Defaults to 1.
        """
        self.map_units = map_units
        self.param_kwargs = {'dtype': dtype, 'device': device}
        self.min_val = min_val
        self.max_val = max_val
        self.self_assoc_weight = self_assoc_weight
        self.mat = self._get_assoc_mat()

    def _get_assoc_mat(self) -> torch.Tensor:
        """Gets an association matrix. The values are randomly generated 
        within the range [min_val, max_val]. The diagonal is set to 
        self_assoc_weight."""
        mat = torch.rand(self.map_units, self.map_units, **self.param_kwargs
                        ) * (self.max_val - self.min_val) + self.min_val
        return mat.fill_diagonal_(self.self_assoc_weight)

    def constrain_(self) -> torch.Tensor:
        """Constrains the association matrix. Sets values outside the range 
        [min_val, max_val] to random values within the range. Sets the 
        diagonal to self_assoc_weight."""
        mask = (self.mat < self.min_val) | (self.mat > self.max_val)
        self.mat[mask] = torch.rand(mask.sum(), **self.param_kwargs) * (
            self.max_val - self.min_val) + self.min_val
        return self.mat.fill_diagonal_(self.self_assoc_weight)

    def __repr__(self) -> str:
        return f'Association Matrix containing:\n{self.mat}'


class ICA(torch.nn.Module):
    """An Inter-City Association layer.

    The ICA layer is a linear layer that uses an association matrix to 
    perform a linear transformation. The association matrix is constrained 
    to have values within the range [min_val, max_val] and the diagonal 
    is set to self_assoc_weight. The association matrix is used to explain 
    the inter-city association between cities.

    .. math::

        \begin{array}{ll}
        y = W_{assoc} x \\
        \end{array}

    Inputs: x
        - **x**: Input of shape `(batch_size, map_units, num_attrs)` or 
            `(map_units, num_attrs)`. Tensor containing the attributes of 
            the cities.
    
    Outputs: y
        - **y**: Output of shape `(batch_size, map_units, num_attrs)` or 
            `(map_units, num_attrs)`. Tensor containing the attributes of 
            the cities. The attributes are transformed by the association 
            of the cities.

    Attributes:
        map_units: The number of map units.
        param_kwargs: The keyword arguments for the parameters.
        assoc_mat: The association matrix.
        w_assoc: The association matrix as a parameter, of shape 
            `(map_units, map_units)`.
        dim_changer: The dimension changer.
    """

    def __init__(self,
                 map_units: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        """Initializes an ICA layer.

        Args:
            map_units: The number of map units.
            dtype: The data type of the map layer. Defaults to None.
            device: The device of the map layer. Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.param_kwargs = {'dtype': dtype, 'device': device}
        self.assoc_mat = AssociationMatrix(self.map_units, **self.param_kwargs)
        self.w_assoc = torch.nn.Parameter(self.assoc_mat.mat)
        self.dim_changer = DimensionChanger(map_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            (x.dim() in (2, 3)) and (x.shape[-2] == self.map_units)
        ), f"""ICA: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), num_attrs) or 2-D of shape (map_units(
            {self.map_units}), num_attrs), but received {x.dim()}-D tensor 
            of shape {x.shape}"""

        self.assoc_mat.constrain_()
        is_batched = x.dim() == 3
        if not is_batched:
            x = torch.unsqueeze(x, 0)
        x = self.dim_changer.three_to_two(x)
        y = torch.matmul(self.w_assoc, x)
        y = self.dim_changer.two_to_three(y)
        return y if is_batched else y.squeeze(0)


class ICFA(torch.nn.Module):
    """An Inter-City Feature Association layer.

    The ICFA layer is a linear layer that uses association matrices to perform 
    a linear transformation. The association matrices are constrained to have 
    values within the range [min_val, max_val] and the diagonal is set to 
    self_assoc_weight. The association matrices are used to explain the 
    inter-city association between the attributes of the cities.

    .. math::

        \begin{array}{ll}
        [x_{attr_1}, x_{attr_2}, ..., x_{attr_n}] = x \\
        y_{attr_n} = W_{assoc_n} x_{attr_n} \\
        y = [y_{attr_1}, y_{attr_2}, ..., y_{attr_n}] \\
        \end{array}
    
    If there are only one attribute, then the ICFA layer is equivalent to the 
    ICA layer.

    Inputs: x
        - **x**: Input of shape `(batch_size, map_units, num_attrs)` or 
            `(map_units, num_attrs)`. Tensor containing the attributes of 
            the cities.
    
    Outputs: y
        - **y**: Output of shape `(batch_size, map_units, num_attrs)` or 
            `(map_units, num_attrs)`. Tensor containing the attributes of 
            the cities. The attributes are transformed by the association 
            of the cities.
    
    Attributes:
        map_units: The number of map units.
        num_attrs: The number of attributes.
        param_kwargs: The keyword arguments for the parameters.
        assoc_mat_{i}: The association matrix for the i-th attribute.
        w_assoc_{i}: The association matrix for the i-th attribute as a 
            parameter, of shape `(map_units, map_units)`.
    """

    def __init__(self, map_units: int, num_attrs: int,
                 dtype: Optional[torch.dtype],
                 device: Optional[torch.device]) -> None:
        """Initializes an ICFA layer.

        Args:
            map_units: The number of map units.
            num_attrs: The number of attributes.
            dtype: The data type of the map layer. Defaults to None.
            device: The device of the map layer. Defaults to None.
        """
        super().__init__()
        self.map_units = map_units
        self.num_attrs = num_attrs
        self.param_kwargs = {'dtype': dtype, 'device': device}
        for idx in range(self.num_attrs):
            assoc_mat = AssociationMatrix(self.map_units, **self.param_kwargs)
            w_assoc = torch.nn.Parameter(assoc_mat.mat)
            setattr(self, f'assoc_mat_{idx}', assoc_mat)
            setattr(self, f'w_assoc_{idx}', w_assoc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert ((x.dim() in (2, 3)) and (
            x.shape[-2:] == (self.map_units, self.num_attrs)
        )), f"""ICFA: Expected x to be 3-D of shape (batch_size, map_units(
            {self.map_units}), num_attrs({self.num_attrs})) or 2-D of shape 
            (map_units({self.map_units}), num_attrs({self.num_attrs})), but 
            received {x.dim()}-D tensor of shape {x.shape}"""
        
        is_batched = x.dim() == 3
        if not is_batched:
            x = torch.unsqueeze(x, 0)
        x = torch.transpose(x, 0, 2)
        for idx in range(self.num_attrs):
            assoc_mat = getattr(self, f'assoc_mat_{idx}')
            assoc_mat.constrain_()
            x[idx] = torch.matmul(getattr(self, f'w_assoc_{idx}'), x[idx])
        x = torch.transpose(x, 0, 2)
        return x if is_batched else x.squeeze(0)