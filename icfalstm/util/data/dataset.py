import datetime
import os
import pickle
from contextlib import suppress
from typing import Literal

import numpy as np
import torch

import icfalstm.util.data.data as data
import icfalstm.util.reader as reader
from icfalstm.util.directory import Directory

__all__ = ['DataDict', 'Dataset']


class DataDict:
    """A class to store the times of input and target by config.

    Attributes:
        dirname (str): The directory name of the data.
        config (reader.Config): The configuration instance.
        setting (reader.Setting): The setting instance.
        file_type (['csv', 'npz']): The extension of the data files.
        data ([data.CSVData, data.NPZData]): The data class.
        data_kwargs (dict): The keyword arguments for the data class.
        base_names (list): The list of base names of the data files.
        input_time_delta_list (list): The list of input time deltas.
        target_time_delta_list (list): The list of target time deltas.
        input_dict (dict): The dictionary of all input data.
        target_dict (dict): The dictionary of all target data.
        train_input_dict (dict): The dictionary of train input data.
        validation_input_dict (dict): The dictionary of validation input data.
        test_input_dict (dict): The dictionary of test input data.
        train_target_dict (dict): The dictionary of train target data.
        validation_target_dict (dict): The dictionary of validation target data.
        test_target_dict (dict): The dictionary of test target data.
        train_length (int): The length of train data.
        validation_length (int): The length of validation data.
        test_length (int): The length of test data.
    """
    # The format of the time string.
    TIME_FORMAT = '%Y%m%d%H%M'

    def __init__(self, dirname: str, config: reader.Config,
                 setting: reader.Setting, file_type: Literal['csv',
                                                             'npz']) -> None:
        """Initializes a DataDict object.

        Args:
            dirname (str): The directory name of the data.
            config (reader.Config): The configuration instance.
            setting (reader.Setting): The setting instance.
            file_type (['csv', 'npz']): The extension of the data files, either 
                'csv' or 'npz'.
        """
        self.dirname = dirname
        self.config = config
        self.setting = setting
        self.file_type = file_type
        if self.file_type == 'csv':
            self.data = data.CSVData
            self.data_kwargs = {'config': self.config}
        elif self.file_type == 'npz':
            self.data = data.NPZData
            self.data_kwargs = {}
        else:
            raise ValueError(f'Unknown file_type: {file_type}')

        self.base_names = self._get_base_names()
        self.input_time_delta_list = self._get_input_time_delta_list()
        self.target_time_delta_list = self._get_target_time_delta_list()
        self.input_dict, self.target_dict = {}, {}
        self._generate_input_and_target_filenames()
        assert self.input_dict.keys() == self.target_dict.keys()
        self.length = len(self.input_dict)

        for dataset_state in ('train', 'validation', 'test'):
            for dataset_type in ('input', 'target'):
                data_dict = self._get_data_dict(dataset_state, dataset_type)
                setattr(self, f'{dataset_state}_{dataset_type}_dict',
                        data_dict)
            assert (getattr(self,
                            f'{dataset_state}_input_dict').keys() == getattr(
                                self, f'{dataset_state}_target_dict').keys())
            setattr(self, f'{dataset_state}_length',
                    len(getattr(self, f'{dataset_state}_input_dict')))

    def _get_idx_start_end(
        self, dataset_state: Literal['train', 'validation',
                                     'test']) -> tuple[int, int]:
        """Gets the start and end indices of the dataset.

        Args:
            dataset_state (['train', 'validation', 'test']): The state of the 
                dataset, either 'train', 'validation' or 'test'.

        Raises:
            ValueError: If the dataset_state is invalid.

        Returns:
            tuple[int, int]: The start and end indices of the dataset.
        """
        if dataset_state == 'train':
            idx_start = 0
            idx_end = self.config.get_proportion('train') * len(self)
        elif dataset_state == 'validation':
            idx_start = self.config.get_proportion('train') * len(self)
            idx_end = (self.config.get_proportion('train') +
                       self.config.get_proportion('validation')) * len(self)
        elif dataset_state == 'test':
            idx_start = (self.config.get_proportion('train') +
                         self.config.get_proportion('validation')) * len(self)
            idx_end = len(self)
        else:
            raise ValueError(f'Unknown dataset_state: {dataset_state}')
        return int(idx_start), int(idx_end)

    def _get_data_dict(
            self, dataset_state: Literal['train', 'validation', 'test'],
            dataset_type: Literal['input', 'target']) -> dict[str, list[str]]:
        """Gets the dictionary of input or target filenames for the given 
        dataset_state.

        Args:
            dataset_state (['train', 'validation', 'test']): The state of the 
                dataset, either 'train', 'validation' or 'test'.
            dataset_type (['input', 'target']): The type of the data, either 
                'input' or 'target'.

        Returns:
            dict[str, list[str]]: The dictionary of input or target filenames.
        """
        data_dict_for_type = getattr(self, f'{dataset_type}_dict')
        idx_start, idx_end = self._get_idx_start_end(dataset_state)
        return {
            idx: data_dict_for_type[key]
            for idx, key in enumerate(range(idx_start, idx_end))
        }

    def __len__(self):
        """Gets the number of input and target filenames.

        Returns:
            int: The number of input and target filenames.
        """
        return self.length

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
            list[list[int]]: The list of target time deltas.
        """
        return [[(input_delta + self.config.prediction_interval + time_delta)
                 for time_delta in range(self.config.prediction_period)]
                for input_delta in range(self.config.input_hours_num)]

    def _get_input_filenames(self, state: datetime.datetime) -> list[str]:
        """Gets the list of input filenames for the given state.

        Args:
            state (datetime.datetime): The start of the times.

        Raises:
            FileNotFoundError: If the data files do not exist for the given
                state.

        Returns:
            list[str]: The list of input filenames for the given state.
        """
        times = [(state + datetime.timedelta(hours=time_delta)).strftime(
            DataDict.TIME_FORMAT) for time_delta in self.input_time_delta_list]
        if not self._has_times_data(times):
            raise FileNotFoundError
        return [f'{time}.{self.file_type}' for time in times]

    def _get_target_filenames(self, state: datetime.datetime) -> list[str]:
        """Gets the list of target filenames for the given state.

        Args:
            state (datetime.datetime): The start of the times.

        Raises:
            FileNotFoundError: If the data files do not exist for the given
                state.

        Returns:
            list[str]: The list of target filenames for the given state.
        """
        times_list = [[
            (state + datetime.timedelta(hours=time_delta)).strftime(
                DataDict.TIME_FORMAT) for time_delta in period_time_delta_list
        ] for period_time_delta_list in self.target_time_delta_list]
        if not all(self._has_times_data(times) for times in times_list):
            raise FileNotFoundError
        return [[f'{time}.{self.file_type}'
                 for time in times]
                for times in times_list]

    def _generate_exist_datadict(self) -> None:
        """Generates the data dictionary for the existing data files.

        Raises:
            FileNotFoundError: If the data dictionary does not exist.
        """
        data_dir = Directory(self.dirname)
        data_dict_foldernames = data_dir.get_exist_folder_for_usage(
            'data_dict')
        for data_dict_foldername in data_dict_foldernames:
            data_dict_dirname = os.path.join(self.dirname,
                                             data_dict_foldername)
            data_dict_config = reader.Config(
                os.path.join(data_dict_dirname, 'config.txt'))
            if self.config.is_equal_to_for_usage(data_dict_config,
                                                 self.setting, 'data_dict'):
                for dataset_type in ('input', 'target'):
                    with open(
                            os.path.join(data_dict_dirname,
                                         f'{dataset_type}_dict.pickle'),
                            'rb') as f:
                        setattr(self, f'{dataset_type}_dict', pickle.load(f))
                return
        raise FileNotFoundError

    def _generate_input_and_target_filenames(self) -> None:
        """Generates the input and target filenames and add them to input_dict 
        and target_dict.
        """
        try:
            self._generate_exist_datadict(self)
        except FileNotFoundError:
            state = self.config.get_time('start')
            end = self.config.get_time('end')
            idx = 0
            while state <= end:
                with suppress(FileNotFoundError):
                    input_filenames = self._get_input_filenames(state)
                    target_filenames = self._get_target_filenames(state)
                    self.input_dict[idx] = input_filenames
                    self.target_dict[idx] = target_filenames
                    idx += 1
                state += datetime.timedelta(hours=1)

    def _get_filnames_data(
            self, filenames: list[str],
            which: Literal['input', 'target']) -> list[np.ndarray]:
        """Gets the data for the given filenames.

        Args:
            filenames (list[str]): The list of filenames.
            which (['input', 'target']): The type of the data, either 'input' 
                or 'target'.

        Returns:
            np.ndarray: The data for the given filenames.
        """
        return [
            self.data(os.path.join(self.dirname, filename),
                      **self.data_kwargs).get_data(which)
            for filename in filenames
        ]

    def get_input_data(self, idx: int,
                       dataset_state: Literal['train', 'validation', 'test'],
                       device: torch.device) -> torch.Tensor:
        """Gets the input data.

        Args:
            idx (int): The index of the data.
            dataset_state (['train', 'validation', 'test']): The state of the 
                dataset, either 'train', 'validation' or 'test'.
            device (torch.device): The device to load the data to.

        Returns:
            torch.Tensor: The input data.
        """
        filenames = getattr(self, f'{dataset_state}_input_dict')[idx]
        data = self._get_filnames_data(filenames, 'input')
        return torch.tensor(data, device=device)

    def get_target_data(self, idx: int,
                        dataset_state: Literal['train', 'validation', 'test'],
                        device: torch.device) -> torch.Tensor:
        """Gets the target data.

        Args:
            idx (int): The index of the data.
            dataset_state (['train', 'validation', 'test']): The state of the 
                dataset, either 'train', 'validation' or 'test'.
            device (torch.device): The device to load the data to.

        Returns:
            torch.Tensor: The target data.
        """
        filenames_list = getattr(self, f'{dataset_state}_target_dict')[idx]
        data = [
            self._get_filnames_data(filenames, 'target')
            for filenames in filenames_list
        ]
        return torch.mean(torch.tensor(data,
                                       device=device,
                                       dtype=torch.float64),
                          dim=1)

    def save(self) -> None:
        """Saves the data dict.
        """
        directory = Directory(self.dirname)
        data_dict_foldername = directory.get_new_foldername('data_dict')
        data_dict_dirname = os.path.join(self.dirname, data_dict_foldername)
        for dataset_type in ('input', 'target'):
            with open(
                    os.path.join(data_dict_dirname,
                                 f'{dataset_type}_dict.pickle'), 'wb') as f:
                pickle.dump(getattr(self, f'{dataset_type}_dict'), f)
        self.config.save_for_usage(data_dict_dirname, self.setting,
                                   'data_dict')


class Dataset(torch.utils.data.Dataset):
    """The dataset class.
    
    Attributes:
        datadict (DataDict): The DataDict instance.
        device (torch.device): The device to load the data to.
        state (['train', 'validation', 'test']): The state of the dataset,
    """

    def __init__(self, datadict: DataDict, device: torch.device) -> None:
        """Initializes a Dataset object.

        Args:
            datadict (DataDict): The DataDict instance.
        """
        super(Dataset, self).__init__()
        self.datadict = datadict
        self.device = device
        self.state = None

    def switch_to(self, state: Literal['train', 'validation', 'test']):
        """Switches to the given state.

        Args:
            state (['train', 'validation', 'test']): The state to switch to, 
                either 'train', 'validation' or 'test'.
        """
        self.state = state

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the data of input and target.

        Args:
            idx (int): The index of the data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The data of input and target.
        """
        basic_kwargs = {
            'idx': idx,
            'dataset_state': self.state,
            'device': self.device
        }
        input_data = self.datadict.get_input_data(**basic_kwargs)
        target_data = self.datadict.get_target_data(**basic_kwargs)
        return input_data, target_data

    def __len__(self) -> int:
        """Gets the number of data of the dataset state.

        Returns:
            int: The number of data.
        """
        return getattr(self.datadict, f'{self.state}_length')