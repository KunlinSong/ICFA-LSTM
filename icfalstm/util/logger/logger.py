import os
from typing import Literal

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import icfalstm.util.reader as reader

__all__ = ['Logger']


class Logger:
    """A class for logging the train process.

    Attributes:
        config (reader.Config): The configuration object.
        config_path (str): The path to the config file.
        latest_model (str): The path to the latest model state dict.
        best_model (str): The path to the best model state dict.
        train_loss (str): The path to the train loss file.
        train_last_prediction_loss (str): The path to the train 
            last prediction loss file.
        validation_loss (str): The path to the validation loss file.
        validation_last_prediction_loss (str): The path to the 
            validation last prediction loss file.
        test_info (str): The path to the test info file.
        writer (SummaryWriter): A tensorboard summary writer.
        best_epoch (int): The best epoch.
        best_loss (float): The best loss.
        start_epoch (int): The start epoch.
    """

    def __init__(self, dirname: str, config: reader.Config) -> None:
        """Initializes a Logger object.

        Args:
            dirname (str): The directory name.
        """
        self.config = config
        self.config_path = os.path.join(dirname, 'config.txt')
        self.latest_model = os.path.join(dirname, 'latest_model.pth')
        self.best_model = os.path.join(dirname, 'best_model.pth')
        self.last_prediction_best_model = os.path.join(
            dirname, 'last_prediction_best_model.pth')
        self.train_loss = os.path.join(dirname, 'train_loss.csv')
        self.train_last_prediction_loss = os.path.join(
            dirname, 'train_last_prediction_loss.csv')
        self.validation_loss = os.path.join(dirname, 'validation_loss.csv')
        self.validation_last_prediction_loss = os.path.join(
            dirname, 'validation_last_prediction_loss.csv')
        self.test_info = os.path.join(dirname, 'test_info.csv')
        self.writer = SummaryWriter(os.path.join(dirname, 'logs'))
        self.best_epoch, self.best_loss = self._get_best('best')
        _, self.last_prediction_best_loss = self._get_best(
            'last_prediction_best')
        self.start_epoch = self._get_start_epoch()

    def add_loss(self,
                 which: Literal['train', 'train_last_prediction', 'validation',
                                'validation_last_prediction'], loss: float,
                 epoch: int) -> None:
        """Adds a loss to the logger.

        Args:
            which (['train', 'train_last_prediction', 'validation', 
                'validation_last_prediction]): The type of loss to add, either
                'train', 'train_last_prediction', 'validation', or 
                'validation_last_prediction'.
            loss (float): The loss to add.
            epoch (int): The epoch of the loss.
        """
        with open(getattr(self, f'{which}_loss'), 'a') as f:
            if epoch == 0:
                f.write('epoch,loss\n')
            f.write(f'{epoch},{loss}\n')
        self.writer.add_scalar(f'{which}_loss', loss, epoch)

    def _get_start_epoch(self) -> int:
        """Gets the start epoch.

        Returns:
            int: The start epoch.
        """
        if os.path.exists(self.train_loss):
            df = pd.read_csv(self.train_loss)
            return df['epoch'].max() + 1
        return 0

    def _get_best(
            self, which: Literal['best',
                                 'last_prediction_best']) -> tuple[int, float]:
        """Gets the best epoch and loss.

        Args:
            which (['best', 'last_prediction_best']): The type of best to get, 
                either 'best' or 'last_prediction_best'.

        Returns:
            tuple[int, float]: The best epoch and loss.
        """
        loss_file = (self.validation_loss if which == 'best' else
                     self.validation_last_prediction_loss)
        if os.path.exists(loss_file):
            df = pd.read_csv(loss_file)
            filtered_df = df[df['loss'] == df['loss'].min()]
            max_epoch_idx = filtered_df['epoch'].idxmax()
            result = filtered_df.loc[max_epoch_idx, ['epoch', 'loss']]
            return result['epoch'], result['loss']
        return 0, float('inf')

    def load_state_dict(self, which: Literal['best', 'latest'],
                        device: torch.device):
        """Loads the state dict of the model.

        Args:
            which (['best', 'latest']): The type of model to load, either 
                'best' or 'latest'.
            device (torch.device): The device to load the model to.
        """
        return torch.load(getattr(self, f'{which}_model'), map_location=device)

    def save_state_dict(self, model: torch.nn.Module,
                        which: Literal['best', 'last_prediction_best', 'latest']) -> None:
        """Saves the state dict of the model.

        Args:
            model (torch.nn.Module): The model to save.
            which (['best', 'last_prediction_best', 'latest']): The type of model to save, either 
                'best' or 'latest'.
        """
        torch.save(model.state_dict(), getattr(self, f'{which}_model'))

    def _change_best(self, epoch: int, loss: float, which: Literal['best', 'last_prediction_best']) -> None:
        """Changes the best epoch and loss.

        Args:
            epoch (int): The new best epoch.
            loss (float): The new best loss.
            which (['best', 'last_prediction_best']): The type of best to 
                change, either 'best' or 'last_prediction_best'.
        """
        if which == 'best':
            self.best_epoch = epoch
            self.best_loss = loss
        else:
            self.last_prediction_best_loss = loss

    def is_best(self, epoch: int, loss: float,
                which: Literal['best', 'last_prediction_best']) -> bool:
        """Checks if the loss is the best.

        Args:
            epoch (int): The epoch of the loss.
            loss (float): The loss.
            which (['best', 'last_prediction_best']): The type of best to 
                check, either 'best' or 'last_prediction_best'.

        Returns:
            bool: True if the loss is the best, False otherwise.
        """
        best_loss = (self.best_loss
                     if which == 'best' else self.last_prediction_best_loss)
        if loss < best_loss:
            self._change_best(epoch, loss, which)
            return True
        return False

    def add_test_target_prediction(self, target: torch.Tensor,
                                   prediction: torch.Tensor,
                                   step: int) -> None:
        """Adds the target and prediction to the test target prediction file.

        Args:
            target (torch.Tensor): Target values.
            prediction (torch.Tensor): Predicted values.
            step (int): Prediction step.
        """
        for city_idx, (city_target,
                       city_prediction) in enumerate(zip(target, prediction)):
            city_name = self.config.cities[city_idx]
            self.writer.add_scalars(f'{city_name}', {
                'prediction': city_prediction,
                'target': city_target
            }, step)

    def add_graph(self, model: torch.nn.Module) -> None:
        """Adds the model graph to the tensorboard.

        Args:
            model (torch.nn.Module): The model to add.
            input (torch.Tensor): The input to the model.
        """
        fake_input = torch.randn(
            (self.config.input_hours_num, self.batch_size,
             len(self.config.cities), len(self.config.attributes)))
        self.writer.add_graph(model, fake_input)

    def early_stopping(self, epoch: int) -> bool:
        return (epoch > self.config.early_stopping_start) and (
            epoch - self.best_epoch > self.config.early_stopping_patience)

    # TODO: Add the method to add the test info, including mae, rmse and r2 for
    #  each city.
