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
        """Initializes the data object by retrieving input and target data.
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

    def __init__(self, path: str, config: util.Config) -> None:
        self.df = pd.read_csv(path, index_col=0)
        self.cities = config.cities
        self.attributes = config.attributes
        self.targets = config.targets
        super(CSVData, self).__init__()

    def _get_array(self, idx: Union[list, tuple, None],
                   cols: Union[list, tuple, None]) -> np.ndarray:
        if idx is None or cols is None:
            raise ValueError(
                'An error occurred because idx or cols is None.'
                f' idx: {idx}, cols: {cols}'
            )
        else:
            return np.array(self.df.loc[idx, cols])

    def _get_input_data(self) -> None:
        try:
            return self._get_array(idx=self.cities, cols=self.attributes)
        except ValueError as err:
            raise ValueError(
                'The attributes in config file is None. Please add at least'
                ' 1 attribute that exists in the csv data.'
            ) from err

    def _get_target_data(self) -> None:
        try:
            return self._get_array(idx=self.cities, cols=self.targets)
        except ValueError as err:
            raise ValueError(
                'The targets in config file is None. Please add at least'
                ' 1 target that exists in the csv data.'
            ) from err
