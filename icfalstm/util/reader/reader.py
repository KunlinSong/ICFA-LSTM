import os
import datetime
from typing import Union, Literal
from collections.abc import Iterable


class Reader:
    """A class for reading text files."""

    def __init__(self, path) -> None:
        """Initializes the reader class.

        Args:
            path (str): The path of the text.

        Raises:
            FileNotFoundError: If the text does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'The config file not found: {path}')
        self._get_attrs(path)

    @staticmethod
    def _split_key_value(cont: str) -> tuple[str]:
        """Splits content in a line of text into a key-value pair.

        Args:
            cont (str): A content in a line of text containing a key and 
                value separated by a colon.

        Returns:
            tuple[str]: A tuple containing the key-value pair.
        
        Raises:
            ValueError: If the cont does not contain a colon.
        """
        k, v = cont.split(':', 1)
        return k.strip(), v.strip()

    @staticmethod
    def _try_convert_to_number(string: str) -> Union[int, float, str]:
        """Attempts to convert a string to a number.

        Args:
            string (str): The string to convert.

        Returns:
            Union[int, float, str]: If the string can be converted to 
                an integer, returns an integer. If the string can be 
                converted to a float, returns a float. Otherwise, 
                returns the original string.
        """
        for convert_func in (int, float):
            try:
                return convert_func(string)
            except ValueError:
                continue
        return string

    def _process_value(
        self, value: str
    ) -> Union[list[Union[int, float, str]], int, float, str, None]:
        """Processes a value and converts it to an appropriate type.

        Args:
            value (str): The value to process.

        Returns:
            Union[list[Union[int, float, str]], int, float, str, None]: 
                If the value is empty returns None. If the value 
                contains commas, splits it and converts it to a list. 
                Otherwise, attempts to convert it to a number.
        """
        if not value:
            return None
        elif ',' in value:
            value = value.split(',')
            value = [
                self._try_convert_to_number(element.strip())
                for element in value
            ]
        else:
            value = self._try_convert_to_number(value)
        return value

    def _get_attrs(self, path: str) -> None:
        """Reads attributes from the text and sets them on the instance.

        Args:
            path (str): The path of the configuration file.
        """
        with open(path, 'r') as f:
            for line in f:
                cont = line.strip()
                if cont and not cont.startswith('#'):
                    k, v = self._split_key_value(cont)
                    v = self._process_value(v)
                    setattr(self, k, v)


class Setting(Reader):
    """A class for reading setting files."""

    def __init__(self, path) -> None:
        """Initializes the Setting class.

        Args:
            path (str): The path of the setting file.

        Raises:
            FileNotFoundError: If the setting file does not exist.
        """
        super(Setting, self).__init__(path)

    def get_usage_keys(self, usage: Literal['data', 'model']) -> Iterable:
        """Gets the usage attribute of the Setting instance and returns 
        an iterable based on its type.

        Args:
            usage (Literal['data', 'model']): The name of the usage attribute 
                to get.

        Returns:
            Iterable: An iterable containing the info of the usage 
                attribute of the Setting instance.
        """
        keys = getattr(self, usage)
        if keys is None:
            return []
        elif isinstance(keys, str):
            return [keys]
        elif isinstance(keys, Iterable):
            return keys
        else:
            return [str(keys)]


class Config(Reader):
    """A class for reading configuration files."""

    def __init__(self, path: str) -> None:
        """Initializes the Config class.

        Args:
            path (str): The path of the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        super(Config, self).__init__(path)

    def is_equal_to_in_usage(self, other: 'Config', setting: Setting,
                             usage: Literal['data', 'model']) -> bool:
        """Determines if two configuration instances are equal 
        for a given usage.

        Args:
            other (Config): Another configuration instance.
            setting (Setting): A setting instance.
            usage (Literal['data', 'model']): The name of the usage.

        Returns:
            bool: True if the two configuration instances are equal 
                for the given usage, False otherwise.
        """
        keys = setting.get_usage_keys(usage)
        return all(getattr(self, key) == getattr(other, key) for key in keys)

    @staticmethod
    def _to_str(
        data_to_convert: Union[Iterable[Union[int, float, str]], int, float,
                               str, None]
    ) -> str:
        """Converts data to a string.

        Args:
            data_to_convert (Union[Iterable[Union[int, float, str]], 
                int, float, str, None]): The data to convert.

        Returns:
            str: A string representation of the data.
        """
        if data_to_convert is None:
            return ''
        elif isinstance(data_to_convert, str):
            return data_to_convert
        elif isinstance(data_to_convert, Iterable):
            return ','.join(map(str, data_to_convert))
        else:
            return str(data_to_convert)

    def save_for_usage(self, dirname: str, setting: Setting,
                       usage: Literal['data', 'model'], **kwargs) -> None:
        """Saves the configuration for a given usage.

        Args:
            dirname (str): The directory in which to save the 
                configuration file.
            setting (Setting): A setting instance.
            usage (Literal['data', 'model']): The name of the usage.
        
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        keys = setting.get_usage_keys(usage)
        with open(os.path.join(dirname, 'config.txt'), 'w') as f:
            f.writelines((
                f'{key} : {self._to_str(getattr(self, key))}' for key in keys))
            f.writelines((f'{key} : {value}' for key, value in kwargs.items()))

    def get_time(self, which: Literal['start', 'end']) -> datetime.datetime:
        """Gets the start or end time of the configuration.

        Args:
            which (Literal['start', 'end']): The type of time to get.

        Returns:
            datetime.datetime: The start or end time of the configuration.
        """
        return datetime.datetime(year=getattr(self, f'{which}_year'),
                                 month=getattr(self, f'{which}_month'),
                                 day=getattr(self, f'{which}_day'))

    def get_proportion(
            self, which: Literal['training', 'validation',
                                 'testing']) -> float:
        """Gets the training, validation, or testing proportion.

        Args:
            which (Literal['training', 'validation', 'testing']): The 
                type of proportion to get.

        Returns:
            float: The training, validation, or testing proportion.
        """
        if which == 'training':
            idx = 0
        elif which == 'validation':
            idx = 1
        elif which == 'testing':
            idx = 2
        else:
            raise ValueError(f'Invalid proportion type: {which}')
        return self.training_validation_testing[idx] / sum(
            self.training_validation_testing)
