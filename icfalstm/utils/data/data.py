import os

import numpy as np
import pandas as pd

from icfalstm.types import *
from icfalstm.utils.reader.reader import Config


class CSVData:
    """A csv file reader that generates numpy arrays for input and target data.
    
    Attributes:
        path (str): Path to the csv file.
        cities (list): List of cities.
        attributes (list): List of attributes.
        targets (list): List of targets.
        df (pandas.DataFrame): Pandas DataFrame containing the data.
        input_data (numpy.ndarray): Numpy array containing the input data.
        target_data (numpy.ndarray): Numpy array containing the target data.
    """
    # The following attributes are used to specify the rows or columns to be 
    # read from the csv file.
    ATTRIBUTES = ('cities', 'attributes', 'targets')
    def __init__(self, path: str, config: Config) -> None:
        """Initializes the csv file reader.

        Args:
            path (str): Path to the csv file.
            config (Config): Config object containing the attributes to read.
        """
        self.path = path
        for attr in CSVData.ATTRIBUTES:
            setattr(self, attr, config[attr])
        self.df = pd.read_csv(self.path, index_col=0)

    def __getitem__(self, key:str) -> np.ndarray:
        return self.get_data(key)
    
    def get_data(self, which: Literal['input', 'target']) -> np.ndarray:
        """Returns the input or target data as a numpy array.

        Args:
            which (['input', 'target']): Which data to return.

        Raises:
            ValueError: If the argument is invalid.

        Returns:
            np.ndarray: Numpy array containing the input or target data.
        """
        if which == 'input':
            indices = (self.cities, self.attributes)
        elif which == 'target':
            indices = (self.cities, self.targets)
        else:
            raise ValueError(f'Invalid argument: {which}')
        return self.df.loc[*indices].values
    
    def _get_basename_without_ext(self) -> str:
        """Returns the basename of the csv file without the extension.

        Returns:
            str: Basename of the csv file without the extension.
        """
        basename = os.path.basename(self.path)
        return os.path.splitext(basename)[0]

    def to_npz(self, dirname: str) -> None:
        """Saves the input and target data as a numpy archive in the given 
        directory.

        Args:
            dirname (str): The directory to save the numpy archive to.
        """
        basename_no_ext = self._get_basename_without_ext()
        np.savez(
            os.path.join(dirname, f'{basename_no_ext}.npz'),
            input_data=self.get_data('input'),
            target_data=self.get_data('target'),
        )
    
class NPZData:
    """A numpy archive reader that reads the input and target data from a numpy 
    archive.
    
    Attributes:
        path (str): Path to the numpy archive.
        data (numpy.lib.npyio.NpzFile): Numpy archive.
        input_data (numpy.ndarray): Numpy array containing the input data.
        target_data (numpy.ndarray): Numpy array containing the target data.
    """
    def __init__(self, path: str) -> None:
        """Initializes the numpy archive reader.

        Args:
            path (str): Path to the numpy archive.
        """
        self.path = path
        self.data = np.load(self.path)
    
    def __getitem__(self, key:str) -> np.ndarray:
        return self.get_data(key)

    def get_data(self, which: Literal['input', 'target']) -> np.ndarray:
        """Returns the input or target data as a numpy array.

        Args:
            which (['input', 'target']): Which data to return.

        Returns:
            np.ndarray: Numpy array containing the input or target data.
        """
        return self.data[f'{which}_data']