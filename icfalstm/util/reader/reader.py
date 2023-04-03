import os
import datetime
from typing import Union, Literal
from collections.abc import Iterable
from contextlib import suppress


class Reader:
    """A class for reading text files.
    
    Attributes:
        path (str): The path of the text.
    """

    def __init__(self, path: str) -> None:
        """Initializes the reader class.

        Args:
            path (str): The path of the text.

        Raises:
            FileNotFoundError: If the text does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'The file: {path} does not exist.')
        self._get_attrs(path)

    @staticmethod
    def _split_key_value(cont: str) -> tuple[str, str]:
        """Splits a string into a key and value.

        Args:
            cont (str): The string to split.
        
        Returns:
            tuple[str, str]: A tuple containing the key and value.
        """
        k, v = cont.split(':', 1)
        return k.strip(), v.strip()

    @staticmethod
    def _try_convert_to_number(string: str) -> Union[int, float, str]:
        """Attempts to convert a string to a number.

        Args:
            string (str): The string to convert.
        
        Returns:
            Union[int, float, str]: The converted number or the string.
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
        """Processes a value.

        Args:
            value (str): The value to process.
        
        Returns:
            Union[list[Union[int, float, str]], int, float, str, None]: Return 
                None if the value is None.If the value is a list type string, 
                it will be converted to a list. If the info in the value is a 
                number, it will be converted to an int or float.
        """
        if not value:
            return None
        elif ',' in value:
            return [
                self._try_convert_to_number(info.strip())
                for info in value.split(',')
            ]
        else:
            return self._try_convert_to_number(value)

    def _get_attrs(self, path: str) -> None:
        """Gets the attributes of the class from the text file.

        Args:
            path (str): The path of the text file.
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
        """Gets the usage attribute of the Setting instance.

        Args:
            usage (['data', 'model']): The name of the usage attribute to get, 
             either 'data' or 'model'.

        Returns:
            Iterable: The usage attribute of the Setting instance.
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

    def is_equal_to_for_usage(self, other: 'Config', setting: Setting,
                              usage: Literal['data', 'model']) -> bool:
        """Checks if the two configuration instances are equal for a given 
        usage.

        Args:
            other (Config): The other configuration instance.
            setting (Setting): A setting instance.
            usage (['data', 'model']): The name of the usage, either 'data' or 
                'model'.

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
            usage (['data', 'model']): The name of the usage, 'either 'data' or 
                'model'.
            **kwargs: The keyword arguments to save.
        """
        keys = setting.get_usage_keys(usage)
        with open(os.path.join(dirname, 'config.txt'), 'w') as f:
            f.writelines((
                f'{key} : {self._to_str(getattr(self, key))}' for key in keys))
            f.writelines((f'{key} : {value}' for key, value in kwargs.items()))

    def get_time(self, which: Literal['start', 'end']) -> datetime.datetime:
        """Gets the start or end time of the data by the configuration.

        Args:
            which (['start', 'end']): The type of time to get, either 'start' 
                or 'end'.

        Returns:
            datetime.datetime: The start or end time of the data.
        """
        return datetime.datetime(year=getattr(self, f'{which}_year'),
                                 month=getattr(self, f'{which}_month'),
                                 day=getattr(self, f'{which}_day'))

    def get_proportion(self, which: Literal['train', 'validation',
                                            'test']) -> float:
        """Gets the train, validation, or test proportion.

        Args:
            which (['train', 'validation', 'test']): The 
                type of proportion to get, either 'train', 'validation', or 
                'test'.

        Returns:
            float: The train, validation, or test proportion.
        """
        if which == 'train':
            idx = 0
        elif which == 'validation':
            idx = 1
        elif which == 'test':
            idx = 2
        else:
            raise ValueError(f'Invalid proportion type: {which}')
        return self.train_validation_test[idx] / sum(
            self.train_validation_test)
