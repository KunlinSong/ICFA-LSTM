import torch

import icfalstm.nn.layer as mylayer
from icfalstm.types import *


class RNNBase(torch.nn.Module):
    """A basic RNN model.
    
    The model is composed of a normalization layer, an association layer, a RNN 
    layer, and a dense layer. The normalization layer is used to normalize the 
    input to the range of [-1, 1] or [0, 1]. The association layer is a 
    optional layer that is used to associate the input by using the 
    association matrix. The RNN layer is used to process the input. The dense 
    layer is used to output the result.

    Attributes:
        mode (str): The mode of the model.
        map_units (int): The number of units in map.
        in_features (int): The number of input features.
        hidden_units (int): The number of hidden units.
        out_features (int): The number of output features.
        dtype (torch.dtype): The data type of the model.
        device (torch.device): The device of the model.
        norm (torch.nn.Module): The normalization layer.
        assoc (torch.nn.Module): The association layer.
        rnn (torch.nn.Module): The RNN layer.
        dense (torch.nn.Module): The dense layer.
    """

    def __init__(self,
                 mode: Literal['ICFA-LSTM', 'ICA-LSTM', 'LSTM', 'ICFA-GRU',
                               'ICA-GRU', 'GRU', 'ICFA-RNN', 'ICA-RNN', 'RNN'],
                 map_units: int,
                 in_features: int,
                 hidden_units: int,
                 out_features: int,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 batch_first: bool=True) -> None:
        """initializes a basic RNN model.
        
        Args:
            mode (str): The mode of the model.
            map_units (int): The number of units in map.
            in_features (int): The number of input features.
            hidden_units (int): The number of hidden units.
            out_features (int): The number of output features.
            dtype (torch.dtype): The data type of the model. Defaults to None.
            device (torch.device): The device of the model. Defaults to None.
        """
        super().__init__()
        self.mode = mode
        self.map_units = map_units
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.batch_first = batch_first
        self.state = None
        self._init_nn()

    def _get_mode(self):
        return (self.mode.split('-') if '-' in self.mode else (None, self.mode))

    def _init_nn(self):
        self.norm = mylayer.ArcTanNorm()
        assoc_mode, rnn_mode = self._get_mode()
        basic_kwargs = {
            'map_units': self.map_units,
            'dtype': self.dtype,
            'device': self.device
        }
        if assoc_mode == 'ICA':
            self.assoc = mylayer.ICA(**basic_kwargs)
        elif assoc_mode == 'ICFA':
            self.assoc = mylayer.ICFA(num_attrs=self.in_features,
                                      **basic_kwargs)
        elif assoc_mode is None:
            self.assoc = None
        else:
            raise ValueError(f'RNNBase: Invalid mode: {self.mode}. The assoc '
                             'mode need to be in [None, "ICA", "ICFA"]')
        self.rnn = getattr(mylayer,
                           f'{rnn_mode}Cell')(input_size=self.in_features,
                                              hidden_size=self.hidden_units,
                                              **basic_kwargs)
        self.dense = mylayer.Dense(in_features=self.hidden_units,
                                   out_features=self.out_features,
                                   **basic_kwargs)

    def forward(self, x: torch.Tensor, clear_state: bool=True) -> torch.Tensor:
        x = x.to(self.device)
        x = torch.transpose(x, 0, 1) if self.batch_first else x
        result = []
        input_times_num = len(x)
        for time_idx in range(input_times_num):
            y = self.norm(x[time_idx])
            if self.assoc is not None:
                y = self.assoc(y)
            self.state = self.rnn(y, self.state)
            y = self.state[0]
            y = self.dense(y)
            y = torch.unsqueeze(y, 0)
            result.append(y)
        self.state = None if clear_state else self.state
        result = torch.cat(result, 0)
        return torch.transpose(result, 0, 1) if self.batch_first else result
