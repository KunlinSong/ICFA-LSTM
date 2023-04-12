import json
import os

from icfalstm.types import *


class Reader:
    """A reader that reads a json file and returns the options as a dictionary.
    
    Attributes:
        path (str): Path to the json file.
        options (dict): Dictionary containing the options.
    """

    def __init__(self, path: str) -> None:
        """Initializes the reader.

        Args:
            path (str): Path to the json file.
        """
        self.path = path
        self.options = self._read()

    def _read(self) -> dict:
        """Reads the json file and returns the options as a dictionary.

        Returns:
            dict: Dictionary containing the options.
        """
        with open(self.path, 'r') as f:
            return json.load(f)

    def __getitem__(self, key) -> Any:
        return self.options[key]


class ConfigSaver(Reader):
    """A reader that reads a json file and returns the config saving settings 
    as a dictionary.

    Attributes:
        path (str): Path to the json file.
        options (dict): Dictionary containing the config saving settings.
    """

    def __init__(self, path: str) -> None:
        """Initializes the config saver.

        Args:
            path (str): Path to the json file.
        """
        super(ConfigSaver, self).__init__(path)


class Config(Reader):
    """A reader that reads a json file and returns the model training options 
    as a dictionary.

    Attributes:
        path (str): Path to the json file.
        options (dict): Dictionary containing the training options.
        config_saver (ConfigSaver): ConfigSaver object.
    """
    STATES = ('train', 'val', 'test')

    def __init__(self, path: str, config_saver: ConfigSaver) -> None:
        """Initializes the config.

        Args:
            path (str): Path to the json file.
            config_saver (ConfigSaver): ConfigSaver object.
        """
        super().__init__(path)
        self.config_saver = config_saver

    def is_equal(self, other: 'Config', usage: Literal['npz_data', 'model',
                                                       'data_dict']) -> bool:
        """Checks if the two configs are equal for the given usage.

        Args:
            other (Config): The other config instance.
            usage (['npz_data', 'model', 'data_dict']): The usage to compare 
                the two configs. Either 'npz_data', 'model' or 'data_dict'.

        Returns:
            bool: True if the two configs are equal for the given usage, 
                False otherwise.
        """
        return all(self[key] == other[key] for key in self.config_saver[usage])

    def _get_config_path(self, dirname: str) -> str:
        """Returns the path to the config file.

        Args:
            dirname (str): The directory to save the config to.

        Returns:
            str: The path to the config file.
        """
        return os.path.join(dirname, 'config.json')

    def save(self, dirname: str, usage: Literal['npz_data', 'model',
                                                'data_dict'],
             **kwargs) -> None:
        """Saves the config to a json file in the given directory for the given 
        usage.

        Args:
            dirname (str): The directory to save the config to.
            usage (['npz_data', 'model', 'data_dict']): The usage to save the 
                config for. Either 'npz_data', 'model' or 'data_dict'.
            **kwargs: Other keyword arguments to be saved to the config.
        """
        options_to_save = {key: self[key] for key in self.config_saver[usage]}
        config_path = self._get_config_path(dirname)
        with open(config_path, 'w') as f:
            json.dump(options_to_save | kwargs, f, indent=4)

    def _get_proportion_idx(self, which: Literal['train', 'val',
                                                 'test']) -> int:
        """Returns the index of the given proportion type.

        Args:
            which (['train', 'val', 'test']): The type of proportion to return.

        Raises:
            ValueError: If the given type is invalid.

        Returns:
            int: The index of the given type.
        """
        try:
            return Config.STATES.index(which)
        except ValueError as err:
            raise ValueError(f'Invalid proportion type: {which}') from err

    def get_proportion(self, which: Literal['train', 'val', 'test']) -> float:
        """Returns the given proportion.

        Args:
            which (['train', 'val', 'test']): The type of proportion to return.
        
        Returns:
            float: The given proportion.
        """
        idx = self._get_proportion_idx(which)
        return self['train_val_test'][idx] / sum(self['train_val_test'])
