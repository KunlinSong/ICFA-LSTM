import os
import datetime
import numpy as np
import pandas as pd
import torch
from typing import Literal
from contextlib import suppress
import icfalstm.util as util


class DataDict:
    """A class to store the times of input and target by config.

    Attributes:
        dirname (str): The directory name of the data.
        config (util.Config): A configuration instance.
        base_names (list[str]): The list of base names of the data files.
        input_dict (dict[str, list[str]]): The dictionary of input times.
        target_dict (dict[str, list[str]]): The dictionary of target times.
        input_time_delta_list (list[int]): The list of input time deltas.
        target_time_delta_list (list[int]): The list of target time deltas.
    """
    # The format of the time string.
    TIME_FORMAT = '%Y%m%d%H%M'

    def __init__(self, dirname: str, config: util.Config) -> None:
        """Initializes a DataDict object.

        Args:
            dirname (str): The directory name of the data.
            config (util.Config): The configuration instance.
        """
        self.dirname = dirname
        self.config = config
        self.base_names = self._get_base_names()
        self.input_dict, self.target_dict = {}, {}
        self.input_time_delta_list = self._get_input_time_delta_list()
        self.target_time_delta_list = self._get_target_time_delta_list()
        self._generate_input_and_target_times()
        assert self.input_dict.keys() == self.target_dict.keys()

    def _has_times_data(self, times: list[str]) -> bool:
        """Checks if the data files exist for the given times.

        Args:
            times (list[str]): The list of times.

        Returns:
            bool: True if the data files exist for the given times.
        """
        return all((time in self.base_names) for time in times)

    def _is_file(self, filename: str) -> bool:
        """Checks if the given filename is a file.

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the given filename is a file.
        """
        return os.path.isfile(os.path.join(self.dirname, filename))

    def _get_base_names(self) -> list[str]:
        """Gets the list of base names of the data files.

        Returns:
            list[str]: The list of base names of the data files.
        """
        filenames = os.listdir(self.dirname)
        return [
            os.path.splitext(filename)[0]
            for filename in filenames
            if self._is_file(filename)
        ]

    def _get_input_time_delta_list(self) -> list[int]:
        """Gets the list of input time deltas.

        Returns:
            list[int]: The list of input time deltas.
        """
        return list(range(self.config.input_hours_num))

    def _get_target_time_delta_list(self) -> list[int]:
        """Gets the list of target time deltas.

        Returns:
            list[int]: The list of target time deltas.
        """
        return [
            self.config.input_hours_num + self.config.prediction_interval +
            time_delta for time_delta in range(self.config.prediction_period)
        ]

    def _get_times(self, state: datetime.datetime,
                   which: Literal['input', 'target']) -> list[str]:
        """Gets the times of input or target.

        Args:
            state (datetime.datetime): The state of the times.
            which (Literal['input', 'target']): The type of data of the times.

        Raises:
            FileNotFoundError: If the data files do not exist for the 
                given times.

        Returns:
            list[str]: The list of times.
        """
        times = [(state + datetime.timedelta(hours=time_delta)).strftime(
            DataDict.TIME_FORMAT)
                 for time_delta in getattr(self, f'{which}_time_delta_list')]
        if not self._has_times_data(times):
            raise FileNotFoundError
        return times

    def _generate_input_and_target_times(self):
        """Generates the input and target times and add them to input_dict and 
        target_dict.
        """
        state = self.config.get_time('start')
        end = self.config.get_time('end')
        while state <= end:
            with suppress(FileNotFoundError):
                input_times = self._get_times(state, 'input')
                target_times = self._get_times(state, 'target')
                self.input_dict[state.strftime(
                    DataDict.TIME_FORMAT)] = input_times
                self.target_dict[state.strftime(
                    DataDict.TIME_FORMAT)] = target_times
            state += datetime.timedelta(hours=1)
