import datetime
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data

import icfalstm.utils.reader.reader as reader
from icfalstm.types import *
from icfalstm.utils.data.data import CSVData, NPZData
from icfalstm.utils.directory.directory import Directory


class DeltaSet:
    """A class that stores the unique values of a nested list.
    
    Attributes:
        delta_set (set): Set containing the unique values of a nested list.
    """

    def __init__(self) -> None:
        """Initializes the delta set."""
        self.delta_set = set()

    def add(self, item: int) -> None:
        """Adds an item to the delta set.

        Args:
            item (int): Item to add.
        """
        self.delta_set.add(item)

    def get_item_from_iter(self, iter_item: Iterable) -> int:
        """Recursively adds the items of an iterable to the delta set, except 
        for strings.

        Args:
            iter_item (Iterable): Iterable to add.

        Returns:
            int: Number of items added.
        """
        for item in iter_item:
            if isinstance(item, Iterable) and (not isinstance(item, str)):
                self.get_item_from_iter(item)
            else:
                self.add(item)

    def __iter__(self):
        yield from self.delta_set


class DataDict:
    """A class that stores the input and target data as dictionaries.

    Attributes:
        input_dict (dict): Dictionary containing the input data.
        target_dict (dict): Dictionary containing the target data.
        config (Config): Config object containing the training options.
        indices (list): List of indices, which are used to split the data into 
            training, validation, and test sets.
        train_input_dict (dict): Dictionary containing the training input data.
        train_target_dict (dict): Dictionary containing the training target
            data.
        val_input_dict (dict): Dictionary containing the validation input data.
        val_target_dict (dict): Dictionary containing the validation target
            data.
        test_input_dict (dict): Dictionary containing the test input data.
        test_target_dict (dict): Dictionary containing the test target data.
    """
    TIME_FORMAT = '%Y%m%d%H%M'

    def __init__(self, input_dict: dict, target_dict: dict,
                 config: reader.Config) -> None:
        """Initializes the data dictionary.

        Args:
            input_dict (dict): Dictionary containing the input data.
            target_dict (dict): Dictionary containing the target data.
            config (reader.Config): Config object containing the training 
                options.
        """
        assert input_dict.keys() == target_dict.keys()
        self.input_dict = input_dict
        self.target_dict = target_dict
        self.config = config
        self.indices = self._get_indices()
        self._get_state_dicts()

    def __len__(self):
        return len(self.input_dict)

    def _get_state_dicts(self):
        """Splits the data into training, validation, and test sets, and 
        stores them in the corresponding attributes.
        """
        for state in reader.Config.STATES:
            start, end = self._get_state_indices(state)
            for dataset_type in ('input', 'target'):
                setattr(self, f'{state}_{dataset_type}_dict',
                        self._get_data_dict(dataset_type, start, end))

    def _proportion_to_idx(self, proportion: float) -> int:
        """Returns the index corresponding to the given proportion.

        Args:
            proportion (float): Proportion of the data.

        Returns:
            int: Index corresponding to the given proportion.
        """
        return round(len(self) * proportion)

    def _get_proportions(self) -> tuple:
        """Returns the proportions of the data for training, validation, and 
        test sets.

        Returns:
            tuple: Proportions of the data for training, validation, and test 
                sets.
        """
        return (0, self.config.get_proportion('train'),
                1 - self.config.get_proportion('test'), 1)

    def _get_indices(self) -> tuple:
        """Returns the indices of the data for training, validation, and test 
        sets.

        Returns:
            tuple: Indices of the data for training, validation, and test sets.
        """
        return tuple(
            self._proportion_to_idx(proportion)
            for proportion in self._get_proportions())

    def _get_state_indices(
            self, state: Literal['train', 'val', 'test']) -> tuple[int, int]:
        """Returns the indices of the data for the given state.

        Args:
            state (['train', 'val', 'test']): State for which the indices are 
                returned.

        Returns:
            tuple[int, int]: Indices of the start and end of the data for the 
                given state.
        """
        idx = reader.Config.STATES.index(state)
        return self.indices[idx], self.indices[idx + 1]

    def _get_data_dict(self, dataset_type: Literal['input', 'target'],
                       start: int, end: int) -> dict:
        """Returns the data dictionary for the given dataset type and indices.

        Args:
            dataset_type (['input', 'target']): Dataset type for which the 
                data dictionary is returned.
            start (int): Index of the start of the data.
            end (int): Index of the end of the data.

        Returns:
            dict: Data dictionary for the given dataset type and indices.
        """
        data_dict_for_type = getattr(self, f'{dataset_type}_dict')
        return {
            idx: data_dict_for_type[key]
            for idx, key in enumerate(range(start, end))
        }

    def save(self, dirname: str) -> None:
        """Saves the data dictionary to the given directory as two pickle 
        files, one for the input data and one for the target data.

        Args:
            dirname (str): Directory in which the data dictionary is saved.
        """
        directory = Directory(dirname)
        data_dict_dir = directory.get_new_usage_dir('data_dict')
        os.makedirs(data_dict_dir)
        for dataset_type in ('input', 'target'):
            data_dict = getattr(self, f'{dataset_type}_dict')
            with open(os.path.join(data_dict_dir, f'{dataset_type}_dict.pkl'),
                      'wb') as f:
                pickle.dump(data_dict, f)
        self.config.save(data_dict_dir, 'data_dict', len_data=len(self))

    @classmethod
    def from_saved(cls, dirname: str, config: reader.Config) -> 'DataDict':
        """Loads the data dictionary from the given directory.

        Args:
            dirname (str): Directory from which the data dictionary is loaded.
            config (reader.Config): Config object containing the training

        Raises:
            FileNotFoundError: If no suitable data dictionary was found under 
                the given directory.

        Returns:
            DataDict: Data dictionary loaded from the given directory.
        """
        directory = Directory(dirname)
        data_dict_basenames = directory.find_usage_basenames('data_dict')
        for basename in data_dict_basenames:
            data_dict_config_path = directory.join(basename, 'config.json')
            if config.is_equal(
                    reader.Config(data_dict_config_path, config.config_saver),
                    'data_dict'):
                kwargs = cls._get_kwargs_from_saved(directory.join(basename),
                                                    config)
                return cls(**kwargs)
        raise FileNotFoundError(
            f'No suitable data dictionary was found under directory {dirname}.'
        )

    @classmethod
    def _get_kwargs_from_saved(cls, dirname: str,
                               config: reader.Config) -> dict:
        """Returns the keyword arguments for the constructor from the given 
        directory.

        Args:
            dirname (str): Directory from which the keyword arguments are 
                returned.
            config (reader.Config): Config object containing the training 
                parameters.

        Returns:
            dict: Keyword arguments for the constructor from the given 
                directory.
        """
        directory = Directory(dirname)
        kwargs = {}
        for dataset_type in ('input', 'target'):
            with open(directory.join(f'{dataset_type}_dict.pkl'), 'rb') as f:
                kwargs[f'{dataset_type}_dict'] = pickle.load(f)
        kwargs['config'] = config
        return kwargs

    @classmethod
    def _get_basenames_times(cls, dirname: str) -> list[datetime.datetime]:
        """Returns the times corresponding to the basenames in the given 
        directory.

        Args:
            dirname (str): Directory from which the basenames are read.

        Returns:
            list[datetime.datetime]: Times corresponding to the basenames in 
                the given directory. 
        """
        basenames = os.listdir(dirname)
        basenames_without_ext = [
            os.path.splitext(basename)[0] for basename in basenames
        ]
        return [
            datetime.datetime.strptime(basename, cls.TIME_FORMAT)
            for basename in basenames_without_ext
        ]

    @staticmethod
    def _get_input_time_deltas(config: reader.Config) -> list[int]:
        """Returns the time deltas for the input data.

        Args:
            config (reader.Config): Config object containing the training 
                parameters.

        Returns:
            list[int]: Time deltas for the input data.
        """
        return list(range(config['input_hours_num']))

    @staticmethod
    def _get_target_time_deltas(config: reader.Config) -> list[list[int]]:
        """Returns the time deltas for the target data.

        Args:
            config (reader.Config): Config object containing the training 
                parameters.

        Returns:
            list[list[int]]: Time deltas for the target data.
        """
        return [
            list(
                range(
                    input_delta + config['prediction_interval'],
                    input_delta + config['prediction_interval'] +
                    config['prediction_period']))
            for input_delta in range(config['input_hours_num'])
        ]

    @staticmethod
    def _get_deltas_set(*args: Iterable) -> DeltaSet:
        """Returns a DeltaSet object from the given arguments.

        Returns:
            DeltaSet: DeltaSet object from the given arguments.
        """
        deltas_set = DeltaSet()
        deltas_set.get_item_from_iter(args)
        return deltas_set

    @classmethod
    def from_data(cls, dirname: str, config: reader.Config) -> 'DataDict':
        """Returns a DataDict object from the given directory.

        Args:
            dirname (str): Directory from which the DataDict object is 
                returned.
            config (reader.Config): Config object containing the training 
                parameters. 

        Returns:
            DataDict: DataDict object from the given directory.
        """
        basenames_times = cls._get_basenames_times(dirname)
        input_time_deltas = cls._get_input_time_deltas(config)
        target_time_deltas = cls._get_target_time_deltas(config)
        return cls._from_times(basenames_times, input_time_deltas,
                               target_time_deltas, config)

    @classmethod
    def _from_times(cls, basenames_times: list[datetime.datetime],
                    input_time_deltas: list[int],
                    target_time_deltas: list[list[int]],
                    config: reader.Config) -> 'DataDict':
        """Returns a DataDict object from the given times.

        Args:
            basenames_times (list[datetime.datetime]): Times corresponding to 
                the basenames in the given directory.
            input_time_deltas (list[int]): Deltas for the input data.
            target_time_deltas (list[list[int]]): Deltas for the target data.
            config (reader.Config): Config object containing the training 
                parameters.

        Returns:
            DataDict: DataDict object from the given times.
        """
        deltas_set = cls._get_deltas_set(input_time_deltas, target_time_deltas)
        state = min(basenames_times)
        end = max(basenames_times)
        input_dict = {}
        target_dict = {}
        idx = 0
        while state <= end:
            times = (state + datetime.timedelta(hours=delta)
                     for delta in deltas_set)
            if all(data_time in basenames_times for data_time in times):
                input_dict[idx] = cls._get_input_times(state,
                                                       input_time_deltas)
                target_dict[idx] = cls._get_target_times(
                    state, target_time_deltas)
                idx += 1
            state += datetime.timedelta(hours=1)
        return cls(input_dict, target_dict, config)

    @classmethod
    def _get_input_times(cls, state: datetime.datetime,
                         time_deltas: list[int]) -> list[str]:
        """Returns the input times str from the given state and time deltas.

        Args:
            state (datetime.datetime): The basic time to add the time deltas to.
            time_deltas (list[int]): The time deltas to add to the state.

        Returns:
            list[str]: The input times str from the given state and time 
                deltas.
        """
        return [(state + datetime.timedelta(hours=delta)).strftime(
            DataDict.TIME_FORMAT) for delta in time_deltas]

    @classmethod
    def _get_target_times(cls, state: datetime.datetime,
                          time_deltas: list[list[int]]) -> list[list[str]]:
        """Returns the target times str from the given state and time deltas.

        Args:
            state (datetime.datetime): The basic time to add the time deltas to.
            time_deltas (list[list[int]]): The time deltas to add to the state.

        Returns:
            list[list[str]]: The target times str from the given state and 
                time deltas.
        """
        return [[(state + datetime.timedelta(hours=delta)).strftime(
            DataDict.TIME_FORMAT)
                 for delta in deltas]
                for deltas in time_deltas]


def _to_dir(lst: list, dirname: str, filetype: Literal['csv', 'npz']) -> None:
    """Converts the given list of str to a list of paths in the given directory.

    Args:
        lst (list): The list to convert.
        dirname (str): The paths in the converted list will be in this 
            directory.
        filetype (['csv', 'npz']): The filetype of the paths in the converted 
            list.
    """
    result = []
    directory = Directory(dirname)
    for item in lst:
        if isinstance(item, list):
            result.append(_to_dir(item, dirname, filetype))
        else:
            result.append(directory.join(f'{item}.{filetype}'))
    return result


class Dataset(data.Dataset):
    """Dataset class.

    Attributes:
        train_input_dict (dict): Dictionary of the training input data.
        train_target_dict (dict): Dictionary of the training target data.
        val_input_dict (dict): Dictionary of the validation input data.
        val_target_dict (dict): Dictionary of the validation target data.
        test_input_dict (dict): Dictionary of the test input data.
        test_target_dict (dict): Dictionary of the test target data.
        config (reader.Config): Config object containing the training
            parameters.
        filetype (['csv', 'npz']): The filetype of the data.
        state (['train', 'val', 'test']): The current state of the dataset.
    """
    # The names of the needed data dictionaries.
    DATA_DICTS = ('train_input_dict', 'train_target_dict', 'val_input_dict',
                  'val_target_dict', 'test_input_dict', 'test_target_dict')

    def __init__(self, data_dict: DataDict, dirname: str,
                 filetype: Literal['csv', 'npz']) -> None:
        """Initializes the Dataset object.

        Args:
            data_dict (DataDict): DataDict object containing the data.
            dirname (str): The directory containing the data.
            filetype (['csv', 'npz']): The filetype of the data.
        """
        super(Dataset, self).__init__()
        for needed_dict in Dataset.DATA_DICTS:
            source_dict = getattr(data_dict, needed_dict)
            self_dict = {
                key: _to_dir(value, dirname, filetype)
                for key, value in source_dict.items()
            }
            setattr(self, needed_dict, self_dict)
        self.config = data_dict.config
        self.filetype = filetype
        self.state = None

    def switch_to(self, state: Literal['train', 'val', 'test']) -> None:
        """Switches the state of the dataset.

        Args:
            state (['train', 'val', 'test']): The new state of the dataset.

        Raises:
            ValueError: If the given state is invalid.
        """
        if state not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid state: {state}')
        self.state = state

    def _get_data(self, which: Literal['input', 'target'],
                  paths: list[str]) -> list[np.ndarray]:
        """Returns the data from the given paths.

        Args:
            which (['input', 'target']): The type of data to get.
            paths (list[str]): The paths to the data.

        Raises:
            ValueError: If the filetype is invalid.

        Returns:
            torch.Tensor: The data from the given paths.
        """
        if self.filetype == 'csv':
            data = [
                CSVData(path, self.config).get_data(which) for path in paths
            ]
        elif self.filetype == 'npz':
            data = [NPZData(path).get_data(which) for path in paths]
        else:
            raise ValueError(f'Invalid filetype: {self.filetype}')
        return data

    def _get_input_data(self, idx: int) -> torch.Tensor:
        """Returns the input data from the given index.

        Args:
            idx (int): The index of the input data to get.

        Returns:
            torch.Tensor: The input data from the given index.
        """
        input_paths = getattr(self, f'{self.state}_input_dict')[idx]
        data = self._get_data('input', input_paths)
        return torch.from_numpy(np.array(data, dtype=np.float64))

    def _get_target_data(self, idx: int) -> torch.Tensor:
        """Returns the target data from the given index.

        Args:
            idx (int): The index of the target data to get.

        Returns:
            list[np.ndarray]: The target data from the given index.
        """
        target_paths_lst = getattr(self, f'{self.state}_target_dict')[idx]
        data = [
            self._get_data('target', target_paths)
            for target_paths in target_paths_lst
        ]
        return torch.mean(torch.from_numpy(np.array(data, dtype=np.float64)),
                          dim=1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state is None:
            raise ValueError('No state was set')
        input_data = self._get_input_data(idx)
        target_data = self._get_target_data(idx)
        return input_data, target_data

    def __len__(self) -> int:
        return len(getattr(self, f'{self.state}_input_dict'))