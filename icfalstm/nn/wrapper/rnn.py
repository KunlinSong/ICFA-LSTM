import torch


import icfalstm.nn.model.rnn as myrnn
from icfalstm.types import *


class RNNWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, batch_first: bool=True) -> None:
        self.model = model
        self.batch_first = batch_first
        self.state = None
    
    def forward(self, x: torch.Tensor, clear_state: bool=True) -> Any:
        x = x.to(self.model.device)
        x = torch.transpose(x, 0, 1) if self.batch_first else x
        result = []
        input_times_num = len(X)
        for time_idx in range(input_times_num):
            y, self.state = self.model(x[time_idx], self.state)
            y = torch.unsqueeze(y, 0)
            result.append(y)
        self.state = None if clear_state else self.state
        result = torch.cat(result, 0)
        return torch.transpose(result, 0, 1) if self.batch_first else result