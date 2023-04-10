import os
import pickle
from collections import defaultdict
from typing import Any, Literal

import pandas as pd
import torch
import torcheval.metrics as metrics
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from icfalstm.utils.directory import Directory
from icfalstm.utils.reader import Config

__all__ = ['Logger']


class ModelLogger:
    """The ModelLogger class is used to log the model.

    Attributes:
        latest_model (str): The path to the latest model.
        best_model (str): The path to the best model.    
    """

    def __init__(self, dirname: str) -> None:
        """Initializes the ModelLogger object.

        Args:
            dirname (str): The directory name.
        """
        directory = Directory(dirname)
        self.latest_model = directory.join('latest_model.pth')
        self.best_model = directory.join('best_model.pth')

    def get_state_dict(self, which: Literal['best', 'latest'],
                       device: torch.device) -> Any:
        """Gets the state dict of the model.

        Args:
            which (['best', 'latest']): Which model to get.
            device (torch.device): The device to load the model to.

        Raises:
            FileNotFoundError: If the model does not exist.

        Returns:
            Any: The state dict of the model to get.
        """
        model_state_dict_path = getattr(self, f'{which}_model')
        if os.path.exists(model_state_dict_path):
            return torch.load(getattr(self, f'{which}_model'),
                              map_location=device)
        else:
            raise FileNotFoundError(f'{model_state_dict_path} does not exist.')

    def save_state_dict(self, state_dict: Any,
                        which: Literal['best', 'latest']) -> None:
        """Saves the state dict of the model.

        Args:
            state_dict (Any): The state dict to save.
            which (['best', 'latest']): Which model to save.
        """
        torch.save(state_dict, getattr(self, f'{which}_model'))


class LossLogger:
    """The LossLogger class is used to log the loss.

    Attributes:
        train_loss (str): The path to the training loss.
        val_loss (str): The path to the validation loss.
        test_predict (str): The path to the test predictions.
        test_info (str): The path to the test information.
    """

    def __init__(self, dirname: str) -> None:
        """Initializes the LossLogger object.

        Args:
            dirname (str): The directory name.
        """
        directory = Directory(dirname)
        self.train_loss = directory.join('train_loss.csv')
        self.val_loss = directory.join('val_loss.csv')
        self.predicted_true = directory.join('predicted_true.pkl')
        self.test_info = directory.join('test_info.pkl')

    def get_history_best(self) -> tuple[int, float]:
        """Gets the best epoch and loss from the history.

        Returns:
            tuple[int, float]: The best epoch and loss.
        """
        if os.path.exists(self.val_loss):
            df = pd.read_csv(self.val_loss)
            best_df = df[df['loss'] == df['loss'].min()]
            max_epoch_idx = best_df['epoch'].idxmax()
            best_row = best_df.loc[max_epoch_idx, ['epoch', 'loss']]
            return best_row['epoch'], best_row['loss']
        else:
            return 0, float('inf')

    def get_start_epoch(self) -> int:
        """Gets the start epoch.

        Returns:
            int: The start epoch.
        """
        if os.path.exists(self.train_loss):
            df = pd.read_csv(self.train_loss)
            return df['epoch'].max() + 1
        else:
            return 1

    def add_loss(self, which: Literal['train', 'val'], loss: float,
                 epoch: int) -> None:
        """Adds the loss to the log file.

        Args:
            which (['train', 'val']): Which loss to add. 
            loss (float): The loss to add.
            epoch (int): The epoch to add.
        """
        df = pd.DataFrame({'epoch': [epoch], 'loss': [loss]})
        df.to_csv(getattr(self, f'{which}_loss'),
                  mode='a',
                  index=False,
                  header=(not os.path.exists(getattr(self, f'{which}_loss'))))

    def mean_absolute_percentage_error(self, y_predict: torch.Tensor,
                                       y_true: torch.Tensor) -> float:
        """Calculates the mean absolute percentage error.

        Args:
            y_predict (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            float: The mean absolute percentage error.
        """
        return torch.mean(torch.abs((y_true - y_predict) / y_true))

    def predicted_true_loader(self):
        with open(self.predicted_true, 'rb') as f:
            predicted_true = pickle.load(f)
            for attr, attr_dict in predicted_true.items():
                for city, city_dict in attr_dict.items():
                    predicted_vals = city_dict['predicted']
                    true_vals = city_dict['true']
                    predicted_vals = torch.tensor(predicted_vals,
                                                  dtype=torch.float64)
                    true_vals = torch.tensor(true_vals, dtype=torch.float64)
                    yield attr, city, predicted_vals, true_vals

    # TODO
    def add_test_info(self) -> None:
        ind_dict = defaultdict(defaultdict(dict))
        for attr, city, predicted_vals, true_vals in self.predicted_true_loader(
        ):
            mae = torch.nn.functional.l1_loss(predicted_vals, true_vals)
            mse = metrics.functional.mean_squared_error(
                predicted_vals, true_vals)
            r2_score = metrics.functional.r2_score(predicted_vals, true_vals)
            mape = self.mean_absolute_percentage_error(predicted_vals,
                                                       true_vals)
            ind_dict[attr][city] = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2_score,
                'mape': mape
            }
        with open(self.test_info, 'wb') as f:
            pickle.dump(ind_dict, f)

    def save_predicted_true(self, predicted_values: torch.Tensor,
                            true_values: torch.Tensor, config: Config) -> None:
        values = [
            torch.unsqueeze(predicted_values, 0),
            torch.unsqueeze(true_values, 0)
        ]
        values = torch.cat(values, dim=0)
        assert values.dim() in (3, 4)
        predicted_true_dict = {}
        if values.dim() == 3:
            values = torch.permute(2, 0, 1)
            attr = config['targets'][0]
            attr_dict = self._get_attr_dict(values)
            predicted_true_dict[attr] = attr_dict

        if values.dim() == 4:
            values = torch.permute(3, 2, 0, 1)
            for attr_idx, attr_values in enumerate(values):
                attr = config['targets'][attr_idx]
                attr_dict = self._get_attr_dict(attr_values)
                predicted_true_dict[attr] = attr_dict

        with open(self.predicted_true, 'wb') as f:
            pickle.dump(predicted_true_dict, f)

    def _get_attr_dict(self, attr_values: torch.Tensor,
                       config: Config) -> dict:
        attr_dict = {}
        for city_idx, city_values in enumerate(attr_values):
            city = config['cities'][city_idx]
            city_dict = self._get_city_dict(city_values)
            attr_dict[city] = city_dict
        return attr_dict

    @staticmethod
    def _get_city_dict(city_values: torch.Tensor) -> dict:
        return {
            'Predicted_Values': city_values[0].cpu().tolist(),
            'True_Values': city_values[1].cpu().tolist()
        }


class Logger:
    """The Logger class is used to log the model.

    Attributes:
        config_path (str): The path to the config file.
        model_logger (ModelLogger): The model logger.
        loss_logger (LossLogger): The loss logger.
        config (Config): The config object.
        writer (SummaryWriter): The tensorboard writer.
        best_epoch (int): The best epoch.
        best_loss (float): The best loss.
        start_epoch (int): The start epoch.
    """

    def __init__(self, dirname: str, config: Config) -> None:
        """Initializes the Logger object.

        Args:
            dirname (str): The directory name.
            config (Config): The config object.
        """
        root_dir = Directory(dirname)
        model_dir = root_dir.join('models')
        logs_dir = root_dir.join('logs')
        self._mk_not_exists(model_dir)
        self._mk_not_exists(logs_dir)
        self.config_path = root_dir.join('config.json')
        self.model_logger = ModelLogger(model_dir)
        self.loss_logger = LossLogger(logs_dir)
        self.config = config
        self.writer = SummaryWriter(root_dir.join('tensorboard_logs'))
        self.best_epoch, self.best_loss = self._get_history_best()
        self.start_epoch = self._get_start_epoch()

    @staticmethod
    def _mk_not_exists(dirname: str) -> None:
        """Creates a directory if it does not exist.

        Args:
            dirname (str): The directory name.
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def _get_history_best(self) -> tuple[int, float]:
        """Gets the best epoch and loss from the history.

        Returns:
            tuple[int, float]: The best epoch and loss.
        """
        return self.loss_logger.get_history_best()

    def _get_start_epoch(self) -> int:
        """Gets the start epoch.

        Returns:
            int: The start epoch.
        """
        return self.loss_logger.get_start_epoch()

    def get_state_dict(self, which: Literal['best', 'latest'],
                       device: torch.device) -> Any:
        """Gets the state dict.

        Args:
            which (['best', 'latest']): Which state dict to get.
            device (torch.device): The device.

        Returns:
            Any: The state dict.
        """
        return self.model_logger.get_state_dict(which, device)

    def save_state_dict(self, state_dict: dict,
                        which: Literal['best', 'latest']) -> None:
        """Saves the state dict.

        Args:
            state_dict (dict): The state dict.
            which (['best', 'latest']): Which state dict to save.
        """
        self.model_logger.save_state_dict(state_dict, which)

    def is_best(self, epoch: int, loss: float) -> bool:
        """Checks if the loss is the best. If it is, it updates the best epoch 
        and loss.

        Args:
            epoch (int): The epoch.
            loss (float): The loss.

        Returns:
            bool: True if the loss is the best, False otherwise.
        """
        if loss < self.best_loss:
            self.best_epoch = epoch
            self.best_loss = loss
            return True
        else:
            return False

    def add_graph(self, model: torch.nn.Module) -> None:
        """Adds the graph to the tensorboard.

        Args:
            model (torch.nn.Module): The model.
        """
        fake_input = torch.randn(self.config['batch_size'],
                                 self.config['input_hours_num'],
                                 len(self.config['cities'],
                                     len(self.config['attributes'])),
                                 dtype=torch.float64,
                                 device=model.device)
        self.writer.add_graph(model, fake_input)

    def early_stopping(self, epoch: int) -> bool:
        """Checks if the early stopping condition is met.

        Args:
            epoch (int): The epoch.

        Returns:
            bool: True if the early stopping condition is met, False otherwise.
        """
        return ((epoch > self.config['early_stopping_start']) and
                ((epoch - self.best_epoch) >
                 self.config['early_stopping_patience']))

    def add_loss(self, which: Literal['train', 'val'], loss: float,
                 epoch: int) -> None:
        """Adds the loss to the tensorboard and the loss logger.

        Args:
            which (['train', 'val']): 
                Which loss to add.
            loss (float): The loss.
            epoch (int): The epoch.
        """
        self.writer.add_scalar(f'{which}_loss', loss, epoch)
        self.loss_logger.add_loss(which, loss, epoch)

    # TODO
    def add_test_info(self, test_predict: torch.Tensor,
                      target: torch.Tensor) -> None:
        """Adds the test info to the tensorboard and the loss logger.

        Args:
            test_predict (torch.Tensor): The predicted values.
            target (torch.Tensor): The true values.
        """
        self.loss_logger.add_test_info(test_predict, target)
        test_predict = test_predict.detach().cpu().numpy()
        target = target.cpu().numpy()
        fig, ax = plt.subplots()
        ax.scatter(target, test_predict)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictied Values')
        ax.set_title('True vs Predicted Values')
        self.writer.add_figure('True vs Predicted Values Scatter Plot', fig)
        x = list(range(len(target)))
        self.writer.add_figure('True vs Predicted Values Line Plot',
                               figure=plt.plot(x, target, label='True Values'),
                               global_step=0)
        self.writer.add_figure('True vs Predicted Values Line Plot',
                               figure=plt.plot(x,
                                               test_predict,
                                               label='Predicted Values'),
                               global_step=0)

    def save_predicted_true(self, predicted_values: torch.Tensor,
                            true_values: torch.Tensor) -> None:
        self.loss_logger.save_predicted_true(predicted_values, true_values,
                                             self.config)
        for attr, city, predicted_vals, true_vals in self.loss_logger.predicted_true_loader(
        ):
            predicted = predicted_vals.numpy()
            true = true_vals.numpy()
            fig, ax = plt.subplots()
            ax.scatter(true, predicted)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictied Values')
            ax.set_title(f'True vs Predicted Values [{attr}, {city}]')
            self.writer.add_figure(
                f'True vs Predicted Values Scatter Plot [{attr}, {city}]', fig)
            x = list(range(len(true_vals)))
            self.writer.add_figure(
                f'True vs Predicted Values Line Plot [{attr}, [{city}]',
                figure=plt.plot(x, true_vals, label='True Values'),
                global_step=0)
            self.writer.add_figure(
                f'True vs Predicted Values Line Plot [{attr}, [{city}]',
                figure=plt.plot(x, predicted_vals, label='Predicted Values'),
                global_step=0)

    def add_test_info(self):
        self.loss_logger.add_test_info()
