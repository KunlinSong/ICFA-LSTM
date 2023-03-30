import os
import numpy as np
import pandas as pd
from typing import Union, Literal
import icfalstm.util as util


class CSVData:
    """A class for reading and processing data from a CSV file.

    Attributes:
        path (str): The path to the CSV file.
        df (pd.DataFrame): The DataFrame containing the data 
            from the CSV file.
        cities (list): The list of cities to use as row indices.
        attributes (list): The list of attributes to use as 
            input columns.
        targets (list): The list of targets to use as output columns.
    """

    def __init__(self, path: str, config: util.Config) -> None:
        """Initializes a CSVData object.

        Args:
            path (str): The path to the CSV file.
            config (util.Config): The configuration object.
        
        Raises:
            ValueError: If any of the required instance variables 
                are None.
        """
        self.path = path
        self.df = pd.read_csv(path, index_col=0)
        self.cities = config.cities
        self.attributes = config.attributes
        self.targets = config.targets
        self._check_vars()
        super(CSVData, self).__init__()

    def _check_vars(self) -> None:
        """Checks if any of the required instance variables are None.

        Raises:
            ValueError: If any of the required instance variables 
                are None.
        """
        for var_name in ('cities', 'attributes', 'targets'):
            if getattr(self, var_name) is None:
                raise ValueError(f'The var: {var_name} is None. '
                                 'Please add at least 1 in config file ')

    def _get_array(self, idx: Union[list, tuple, None],
                   cols: Union[list, tuple, None]) -> np.ndarray:
        """Gets a NumPy array from the DataFrame.

        Args:
            idx (Union[list, tuple, None]): The row indices to get.
            cols (Union[list, tuple, None]): The columns to get.

        Returns:
            np.ndarray: The resulting NumPy array
        """
        return np.array(self.df.loc[idx, cols])

    def get_data(self, which: Literal['input', 'target']) -> np.ndarray:
        """Gets the input or target data as a NumPy array.

        Returns:
            np.ndarray: The input or target data as a NumPy array.
        """
        if which == 'input':
            return self._get_array(idx=self.cities, cols=self.attributes)
        elif which == 'target':
            return self._get_array(idx=self.cities, cols=self.targets)
        else:
            raise ValueError(f'Invalid value for which: {which}')

    def to_npz(self, dirname: str) -> None:
        """Saves the input and target data as a NumPy array.

        Args:
            dirname (str): The directory to save the NumPy array to.
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        base_name = os.path.basename(self.path)
        no_extension = os.path.splitext(base_name)[0]
        npz_path = os.path.join(dirname, f'{no_extension}.npz')
        np.savez(npz_path,
                 input_data=self.get_data('input'),
                 target_data=self.get_data('target'))


class NPZData:
    """A class for reading and processing data from an NPZ file.

    Args:
        npz_file (np.lib.npyio.NpzFile): The NpzFile object 
            containing the data from the NPZ file.
    """

    def __init__(self, path: str) -> None:
        """Initializes an NPZData object.

        Args:
            path (str): The path to the NPZ file.
        """
        self.npz_file = np.load(path)
        super(NPZData, self).__init__()

    def get_data(self, which: Literal['input', 'target']) -> np.ndarray:
        """Gets the input or target data as a NumPy array.

        Returns:
            np.ndarray: The input or target data as a NumPy array.
        """
        if which == 'input':
            return self.npz_file.get('input_data')
        elif which == 'target':
            return self.npz_file.get('target_data')
        else:
            raise ValueError(f'Invalid value for which: {which}')