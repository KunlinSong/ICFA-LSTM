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
        input_dict (dict[str, list[str]]): The dictionary of input filenames.
        target_dict (dict[str, list[str]]): The dictionary of target filenames.
        input_time_delta_list (list[int]): The list of input time deltas.
        target_time_delta_list (list[int]): The list of target time deltas.
    """
    # The format of the time string.
    TIME_FORMAT = '%Y%m%d%H%M'

    def __init__(self, dirname: str, config: util.Config,
                 file_type: Literal['csv', 'npz']) -> None:
        """Initializes a DataDict object.

        Args:
            dirname (str): The directory name of the data.
            config (util.Config): The configuration instance.
            file_type (Literal['csv', 'npz']): The extension of the data files.
        """
        self.dirname = dirname
        self.config = config
        self.file_type = file_type
        self.base_names = self._get_base_names()
        self.input_dict, self.target_dict = {}, {}
        self.input_time_delta_list = self._get_input_time_delta_list()
        self.target_time_delta_list = self._get_target_time_delta_list()
        self._generate_input_and_target_filenames()
        assert self.input_dict.keys() == self.target_dict.keys()
    
    def __len__(self):
        """Gets the number of input and target filenames.

        Returns:
            int: The number of input and target filenames.
        """
        return len(self.input_dict)

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

    def _get_filenames(self, state: datetime.datetime,
                       which: Literal['input', 'target']) -> list[str]:
        """Gets the filenames of input or target.

        Args:
            state (datetime.datetime): The state of the times.
            which (Literal['input', 'target']): The type of data of the times.

        Raises:
            FileNotFoundError: If the data files do not exist for the 
                given times.

        Returns:
            list[str]: The list of filenames of each time.
        """
        times = [(state + datetime.timedelta(hours=time_delta)).strftime(
            DataDict.TIME_FORMAT)
                 for time_delta in getattr(self, f'{which}_time_delta_list')]
        if not self._has_times_data(times):
            raise FileNotFoundError
        return [f'{time}.{self.file_type}' for time in times]

    def _generate_input_and_target_filenames(self) -> None:
        """Generates the input and target filenames and add them to input_dict 
        and target_dict.
        """
        state = self.config.get_time('start')
        end = self.config.get_time('end')
        idx = 0
        while state <= end:
            with suppress(FileNotFoundError):
                input_filenames = self._get_filenames(state, 'input')
                target_filenames = self._get_filenames(state, 'target')
                self.input_dict[idx] = input_filenames
                self.target_dict[idx] = target_filenames
                idx += 1
            state += datetime.timedelta(hours=1)

    def get_data(self, idx: int, which: Literal['input', 'target'],
                 device: torch.device) -> np.ndarray:
        """Gets the data of input or target.

        Args:
            idx (int): The index of the data.
            which (['input', 'target']): The type of data.

        Returns:
            np.ndarray: The data of input or target.
        """
        filenames = getattr(self, f'{which}_dict')[idx]
        data = []
        for filename in filenames:
            filepath = os.path.join(self.dirname, filename)
            if self.file_type == 'csv':
                data.append(
                    util.CSVData(filepath, self.config).get_data(which))
            elif self.file_type == 'npz':
                data.append(util.NPZData(filepath).get_data(which))
            else:
                raise ValueError(f'Invalid file type: {self.file_type}')
        return torch.tensor(data, device=device)


class Dataset(torch.utils.data.Dataset):
    """The dataset class.
    
    Attributes:
        datadict (DataDict): The DataDict instance.
    """

    def __init__(self, datadict: DataDict, device: torch.device) -> None:
        """Initializes a Dataset object.

        Args:
            datadict (DataDict): The DataDict instance.
        """
        super(Dataset, self).__init__()
        self.datadict = datadict
        self.device = device

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the data of input and target.

        Args:
            idx (int): The index of the data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The data of input and target.
        """
        input_data = self.datadict.get_data(idx, 'input')
        target_data = self.datadict.get_data(idx, 'target')
        return input_data, torch.mean(target_data, dim=0)
    
    def __len__(self) -> int:
        """Gets the number of data.

        Returns:
            int: The number of data.
        """
        return len(self.datadict)