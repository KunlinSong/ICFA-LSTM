import datetime
import os
from typing import Literal

__all__ = [
    'TimeRecorder', 'print_divide_line', 'print_epoch_loss', 'print_loss'
]

NUM_SPACE = 2


def print_divide_line() -> None:
    """Prints a divide line.
    """
    print('-' * os.get_terminal_size().columns)


class TimeRecorder:
    """A class to record time.

    Attributes:
        start (datetime.datetime): The start time.
        end (datetime.datetime): The end time.
    """
    FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self) -> None:
        """Initializes a Time object.
        """
        self.start = None
        self.end = None

    def update_time(self, which: Literal['start', 'end']) -> None:
        """Updates the time.

        Args:
            which (['start', 'end']): The time to update.
        """
        setattr(self, which, datetime.datetime.now())

    def get_spend(self) -> None:
        """Gets the time spend.
        """
        return f'{self.end.timestamp() - self.start.timestamp():.2f}'

    def get_time(self, which: Literal['start', 'end']) -> str:
        """Gets the time.

        Args:
            which (['start', 'end']): The time to get.

        Returns:
            str: The time.
        """
        return getattr(self, which).strftime(TimeRecorder.FORMAT)

    def print_start(self) -> None:
        """Print the start time.
        """
        print(f'{" " * NUM_SPACE}began at {self.get_time("end")}.')

    def print_end(self) -> None:
        """Print the end time.
        """
        print(f'{" " * NUM_SPACE}ended at {self.get_time("start")}.')

    def print_spend(self) -> None:
        """Print the spend time.
        """
        print(f'{" " * NUM_SPACE}took {self.get_spend()} seconds.')


def print_batch_loss(loss: float, batch_idx: int, batch_num: int) -> None:
    """Prints the batch loss.

    Args:
        loss (float): The loss.
    """
    print(
        f'\r{" " * NUM_SPACE * 2}batch: {batch_idx} / {batch_num} '
        f'loss: {loss}',
        end='')


def print_epoch_loss(epoch_loss: float) -> None:
    """Prints the epoch loss.

    Args:
        loss (float): The loss.
    """
    print(f'\r{" " * NUM_SPACE * 2}epoch loss: {epoch_loss}')
