import datetime
import os
import statistics
from typing import Literal

__all__ = ['LossRecorder', 'TimeRecorder', 'print_separator']

NUM_SPACE = 2
space_str = ' ' * NUM_SPACE


class TimeRecorder:
    """A Recorder that records the start and end time of a process.

    Attributes:
        start (datetime.datetime): Start time.
    """
    # The format of the time string.
    FORMAT = '%Y-%m-%d %H:%M:%S'
    ATTRIBUTES = ['start', 'end']

    def __init__(self) -> None:
        """Initializes the TimeRecorder object."""
        self.start = datetime.datetime.now()
        self.end = datetime.datetime.now()

    @staticmethod
    def _check_which_arg(arg: str) -> None:
        """Checks if the argument is valid. If not, raises a ValueError.

        Args:
            arg (str): The argument to check.

        Raises:
            ValueError: If the argument is invalid.
        """
        if arg not in TimeRecorder.ATTRIBUTES:
            raise ValueError(f'Invalid argument: {arg}')

    def update_time(self, which: Literal['start', 'end']) -> None:
        """Updates the start or end time.

        Args:
            which (['start', 'end']): The time to update.
        """
        self._check_which_arg(which)
        setattr(self, which, datetime.datetime.now())

    def get_time(self, which: Literal['start', 'end']) -> str:
        """Returns the start or end time.

        Args:
            which (['start', 'end']): The time to return.

        Returns:
            str: The start or end time.
        """
        self._check_which_arg(which)
        return getattr(self, which).strftime(TimeRecorder.FORMAT)

    def get_spend_seconds(self) -> str:
        """Returns the spend time in seconds.

        Returns:
            str: The spend time in seconds.
        """
        return f'{self.end.timestamp() - self.start.timestamp():.2f}'

    def print_time(self, which: Literal['start', 'end']) -> None:
        """Prints the start or end time.

        Args:
            which (['start', 'end']): The time to print.
        """
        self._check_which_arg(which)
        print(f'{space_str}{which.capitalize()} time: {self.get_time(which)}')

    def print_spend(self) -> None:
        """Prints the spend time."""
        print(f'{space_str}Spend time: {self.get_spend_seconds()}')


class BatchLoss:
    """A recorder that records the info of batches in a epoch, including batch 
    index, batch number, and loss.
    
    Attributes:
        batch_num (int): The number of batches in a epoch.
        batch_idx (int): The index of the current batch.
        loss (float): The loss of the current batch.
    """

    def __init__(self) -> None:
        """Initializes the BatchLoss object."""
        self.batch_num = 0
        self.batch_idx = 0
        self.loss = float('inf')

    def new_epoch(self, batch_num: int) -> None:
        """Updates the batch number for a new epoch.

        Args:
            batch_num (int): The number of batches in a epoch.
        """
        self.batch_num = batch_num

    def update(self, batch_idx: int, loss: float) -> None:
        """Updates the batch index and loss of the current batch.

        Args:
            batch_idx (int): The index of the current batch.
            loss (float): The loss of the current batch.
        """
        self.batch_idx = batch_idx
        self.loss = loss

    def __repr__(self) -> str:
        return (f'\r{space_str * 2}Batch: {self.batch_idx} / {self.batch_num} '
                f'Loss: {self.loss:10.4g}')


class EpochLoss:
    """A recorder that records the info of a epoch, including epoch index, 
    epoch number, and loss.

    Attributes:
        max_epoch (int): The number of epochs.
        epoch_idx (int): The index of the current epoch.
        epoch_loss_lst (list[float]): The list of losses in the current epoch.
    """

    def __init__(self, max_epoch: int) -> None:
        """Initializes the EpochLoss object 

        Args:
            max_epoch (int): The number of epochs.
        """
        self.max_epoch = max_epoch
        self.epoch_idx = 0
        self.epoch_loss_lst = []

    def new_epoch(self, epoch_idx: int) -> None:
        """Updates the epoch index for a new epoch.

        Args:
            epoch_idx (int): The index of the current epoch.
        """
        self.epoch_idx = epoch_idx
        self.epoch_loss_lst = []

    def update(self, loss: float) -> None:
        """Updates the loss of the current batch.

        Args:
            loss (float): The loss of the current batch.
        """
        self.epoch_loss_lst.append(loss)

    def get_epoch_loss(self) -> float:
        """Returns the mean loss of the current epoch.

        Returns:
            float: The mean loss of the current epoch.
        """
        return statistics.mean(self.epoch_loss_lst)

    def __repr__(self) -> str:
        return (f'\n{space_str * 2}Loss: {self.get_epoch_loss():>10.4g}')


class LossRecorder:
    """A recorder that records the info of batches and epochs.

    Attributes:
        epoch_loss (EpochLoss): The EpochLoss object.
        batch_loss (BatchLoss): The BatchLoss object.
    """

    def __init__(self, max_epoch: int) -> None:
        """Initializes the LossRecorder object.

        Args:
            max_epoch (int): The number of epochs.
        """
        self.epoch_loss = EpochLoss(max_epoch)
        self.batch_loss = BatchLoss()

    def new_epoch(self, epoch_idx: int, batch_num: int) -> None:
        """Updates the epoch index and batch number for a new epoch.

        Args:
            epoch_idx (int): The index of the current epoch.
            batch_num (int): The number of batches in a epoch.
        """
        self.epoch_loss.new_epoch(epoch_idx)
        self.batch_loss.new_epoch(batch_num)

    def update(self, batch_idx: int, loss: float) -> None:
        """Updates the batch index, loss of the current batch, and loss of the 
        current epoch.

        Args:
            batch_idx (int): The index of the current batch.
            loss (float): The loss of the current batch.
        """
        self.epoch_loss.update(loss)
        self.batch_loss.update(batch_idx, loss)

    def print_batch_loss(self) -> None:
        """Prints the loss of the current batch."""
        print(self.batch_loss, end='')

    def print_epoch_loss(self) -> None:
        """Prints the loss of the current epoch."""
        print(self.epoch_loss)
    
    def get_epoch_loss(self) -> float:
        """Returns the loss of the current epoch."""
        return self.epoch_loss.get_epoch_loss()


def print_separator() -> None:
    """Prints a separator."""
    print('-' * os.get_terminal_size().columns)