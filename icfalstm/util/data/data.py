import os
import numpy as np
import pandas as pd
from typing import Union
import icfalstm.util as util


class Data:
    """A class representing data with input and target values.

    This class is intended to be subclassed and the methods 
    '_get_input_data' and '_get_target_data' should be implemented.
    """

    def __init__(self) -> None:
        """Initializes the data object by retrieving input and 
        target data.
        """
        self.input_data = self._get_input_data()
        self.target_data = self._get_target_data()

    def _get_input_data(self) -> None:
        """Retrieves input data.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented 
                by a subclass.
        """
        raise NotImplementedError

    def _get_target_data(self) -> None:
        """Retrieves target data.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented 
                by a subclass.
        """
        raise NotImplementedError


class CSVData(Data):
    """A class for reading and processing data from a CSV file.

    Attributes:
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
        self.df = pd.read_csv(path, index_col=0)
        self.cities = config.cities
        self.attributes = config.attributes
        self.targets = config.targets
        self._check_vars()
        super(CSVData, self).__init__()

    def _check_vars(self):
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

    def _get_input_data(self) -> np.ndarray:
        """Gets the input data as a NumPy array.

        Returns:
            np.ndarray: The input data as a NumPy array.
        """
        return self._get_array(idx=self.cities, cols=self.attributes)

    def _get_target_data(self) -> np.ndarray:
        """Gets the target data as a NumPy array.

        Returns:
            np.ndarray: The target data as a NumPy array
        """
        return self._get_array(idx=self.cities, cols=self.targets)


class NPZData(Data):
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

    def _get_input_data(self) -> np.ndarray:
        """Gets the input data as a NumPy array.

        Returns:
            np.ndarray: The input data as a NumPy array.
        """
        return self.npz_file.get('input_data')

    def _get_target_data(self) -> np.ndarray:
        """Gets the target data as a NumPy array.

        Returns:
            np.ndarray: The target data as a NumPy array.
        """
        return self.npz_file.get('target_data')