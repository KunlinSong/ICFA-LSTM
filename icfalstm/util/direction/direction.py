import os
import re


class Direction:
    """The class for the direction.

    Attributes:
        dirname (str): The directory name.
    """

    def __init__(self, dirname: str) -> None:
        """Initializes a Direction object.

        Args:
            dirname (str): The directory name.
        """
        self.dirname = dirname

    def _is_usage_dir(self, basename: str, usage: str) -> bool:
        """Checks if the given basename is a directory and starts with the 
        given usage.

        Args:
            basename (str): The basename to check.
            usage (str): The usage to check.

        Returns:
            bool: True if the given basename is a directory and starts with 
                the given usage.
        """
        return (os.path.isdir(os.path.join(self.dirname, basename)) and
                basename.startswith(usage))

    def get_new_foldername(self, usage: str) -> str:
        """Gets the new folder name for the given usage.

        Args:
            usage (str): The usage of the folder.

        Returns:
            str: The new folder name for the given usage.
        """
        exist_folder = [
            filename for filename in os.listdir(self.dirname)
            if self._is_usage_dir(filename, usage)
        ]
        # Use regex to get the number in foldername.
        pattern = r"\d+"
        folder_numbers = {
            re.findall(pattern, folder)[0] for folder in exist_folder
        }
        num = 0
        while num in folder_numbers:
            num += 1
        return f'{usage}_{num}'

    def mkdir(self, foldername: str) -> None:
        """Creates a new folder with the given folder name.

        Args:
            foldername (str): The folder name.
        """
        folder_path = os.path.join(self.dirname, foldername)
        os.mkdir(folder_path)
