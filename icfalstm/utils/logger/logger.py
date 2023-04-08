import os
from typing import Any, Literal

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from icfalstm.utils.directory import Directory


class ModelLogger:
    def __init__(self, dirname: str) -> None:
        directory = Directory(dirname)
        self.latest_model = directory.join('latest_model.pth')
        self.best_model = directory.join('best_model.pth')
        self.last_predict_best_model = directory.join('last_predict_best_model.pth')
    
    
    def get_state_dict(self, which: Literal['best', 'latest'], device: torch.device) -> Any:
        return torch.load(getattr(self, f'{which}_model'), map_location=device)
    
    def save_state_dict(self, state_dict: dict, which: Literal['best', 'latest']) -> None:
        torch.save(state_dict, getattr(self, f'{which}_model'))

class LossLogger:
    def __init__(self, dirname: str) -> None:
        directory = Directory(dirname)
        self.train_loss = directory.join('train_loss.csv')
        self.train_last_predict_loss = directory.join(
            'train_last_predict_loss.csv')
        self.val_loss = directory.join('val_loss.csv')
        self.val_last_predict_loss = directory.join('val_last_predict_loss.csv')
        self.test_info = directory.join('test_info.csv')
    
    def get_history_best(self) -> tuple[int, float]:
        if os.path.exists(self.val_loss):
            df = pd.read_csv(self.val_loss)
            best_df = df[df['loss'] == df['loss'].min()]
            max_epoch_idx = best_df['epoch'].idxmax()
            best_row = best_df.loc[max_epoch_idx, ['epoch', 'loss']]
            return best_row['epoch'], best_row['loss']
        else:
            return 0, float('inf')
    
    def get_start_epoch(self) -> int:
        if os.path.exists(self.train_loss):
            df = pd.read_csv(self.train_loss)
            return df['epoch'].max() + 1
        else:
            return 0

class Logger:

    def __init__(self, dirname: str) -> None:
        root_dir = Directory(dirname)
        model_dir = Directory(root_dir.join('models'))
        logs_dir = Directory(root_dir.join('logs'))
        self.config = root_dir.join('config.json')
        self.model_logger = ModelLogger(model_dir)
        self.loss_logger = LossLogger(logs_dir)
        self.writer = SummaryWriter(root_dir.join('tensorboard_logs'))
        self.best_epoch, self.best_loss = self._get_history_best()
        self.start_epoch = self._get_start_epoch()
    
    def _get_history_best(self) -> tuple[int, float]:
        return self.loss_logger.get_history_best()
    
    def _get_start_epoch(self) -> int:
        return self.loss_logger.get_start_epoch()
    
    def get_state_dict(self, which: Literal['best', 'latest'], device: torch.device) -> Any:
        return self.model_logger.load_state_dict(which, device)
    
    def save_state_dict(self, state_dict: dict, which: Literal['best', 'latest']) -> None:
        self.model_logger.save_state_dict(state_dict, which)
    
    # TODO: tomorrow update.
    def is_best(self, epoch: int, loss: float) -> bool:
        pass
    