import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
import torcheval.metrics as metrics
import torch.utils.tensorboard as tensorboard

from icfalstm.types import *
from icfalstm.utils.directory.directory import Directory
from icfalstm.utils.reader.reader import Config

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

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

    def __init__(self, dirname: str, config: Config) -> None:
        """Initializes the LossLogger object.

        Args:
            dirname (str): The directory name.
        """
        directory = Directory(dirname)
        self.config = config
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
                    predicted_vals = city_dict['Predicted_Values']
                    true_vals = city_dict['True_Values']
                    predicted_vals = torch.tensor(predicted_vals)
                    true_vals = torch.tensor(true_vals)
                    yield attr, city, predicted_vals, true_vals

    def add_test_info(self) -> None:
        ind_dict = defaultdict(lambda: defaultdict(lambda: dict))
        for attr, city, predicted_vals, true_vals in self.predicted_true_loader(
        ):
            mae = torch.nn.functional.l1_loss(predicted_vals, true_vals)
            mse = metrics.functional.mean_squared_error(
                predicted_vals, true_vals)
            r2_score = metrics.functional.r2_score(predicted_vals, true_vals)
            mape = self.mean_absolute_percentage_error(predicted_vals,
                                                       true_vals)
            ind_dict[attr][city] = {
                'mae': mae.item(),
                'mse': mse.item(),
                'r2_score': r2_score.item(),
                'mape': mape
            }
            
        with open(self.test_info, 'wb') as f:
            pickle.dump(default_to_regular(ind_dict), f)

    def save_predicted_true(self, predicted_values: torch.Tensor,
                            true_values: torch.Tensor) -> None:
        values = [
            torch.unsqueeze(predicted_values, 0),
            torch.unsqueeze(true_values, 0)
        ]
        values = torch.cat(values, dim=0)
        assert values.dim() == 4
        predicted_true_dict = {}
        values = torch.permute(values, (3, 2, 0, 1))
        for attr_idx, attr_values in enumerate(values):
            attr = self.config['targets'][attr_idx]
            attr_dict = self._get_attr_dict(attr_values)
            predicted_true_dict[attr] = attr_dict

        with open(self.predicted_true, 'wb') as f:
            pickle.dump(predicted_true_dict, f)

    def _get_attr_dict(self, attr_values: torch.Tensor) -> dict:
        attr_dict = {}
        for city_idx, city_values in enumerate(attr_values):
            city = self.config['cities'][city_idx]
            city_dict = self._get_city_dict(city_values)
            attr_dict[city] = city_dict
        return attr_dict

    @staticmethod
    def _get_city_dict(city_values: torch.Tensor) -> dict:
        return {
            'Predicted_Values': city_values[0].cpu().tolist(),
            'True_Values': city_values[1].cpu().tolist()
        }


class TensorboardLogger:

    def __init__(self, dirname: str, config: Config) -> None:
        self.writer = tensorboard.SummaryWriter(dirname)
        self.config = config

    def add_graph(self, model: torch.nn.Module) -> None:
        fake_input = torch.randn(self.config['batch_size'],
                                 self.config['input_hours_num'],
                                 len(self.config['cities']),
                                 len(self.config['attributes']),
                                 dtype=model.dtype,
                                 device=model.device)
        self.writer.add_graph(model, fake_input)

    def add_loss(self, which: Literal['train', 'val'], loss: float,
                 epoch: int) -> None:
        self.writer.add_scalar(f'{which}_loss', loss, epoch)

    def add_predicted_true(self, predicted_values: torch.Tensor,
                           true_values: torch.Tensor, city: str, attr: str):
        for idx in range(len(predicted_values)):
            self.writer.add_scalars(
                f'Predicted vs True Values [{city}, {attr}]', {
                    'Predicted Values': predicted_values[idx],
                    'True Values': true_values[idx]
                }, idx)


class Logger:

    def __init__(self, dirname: str, config: Config) -> None:
        self.dirname = dirname
        self.config = config
        root_dir = Directory(dirname)
        model_dir = root_dir.join('models')
        logs_dir = root_dir.join('logs')
        tensorboard_dir = root_dir.join('tensorboard_logs')
        self._mk_not_exists(model_dir)
        self._mk_not_exists(logs_dir)
        self.model_logger = ModelLogger(model_dir)
        self.loss_logger = LossLogger(logs_dir, config)
        self.tensorboard_logger = TensorboardLogger(tensorboard_dir, config)
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
        """Adds the model graph to tensorboard.

        Args:
            model (torch.nn.Module): The model.
        """
        self.tensorboard_logger.add_graph(model)

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
        self.loss_logger.add_loss(which, loss, epoch)
        self.tensorboard_logger.add_loss(which, loss, epoch)

    def save_predicted_true(self, predicted_values: torch.Tensor,
                            true_values: torch.Tensor):
        self.loss_logger.save_predicted_true(predicted_values, true_values)
        for attr, city, predicted_values, true_values in self.loss_logger.predicted_true_loader(
        ):
            self.tensorboard_logger.add_predicted_true(predicted_values,
                                                       true_values, city, attr)

    def add_test_info(self):
        self.loss_logger.add_test_info()
