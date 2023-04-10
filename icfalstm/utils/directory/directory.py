import os

__all__ = ['Directory']


class Directory:
    """The Directory class is a wrapper around the os.path module. It is used 
    to create directories and to find the indices of directories with a given 
    usage.
    """

    def __init__(self, dirname: str) -> None:
        """Initializes the Directory object.

        Args:
            dirname (str): The directory name.
        """
        self.dirname = dirname

    def join(self, *args) -> str:
        """Joins the directory name with one or more pathname components.

        Returns:
            str: The joined path.
        """
        return os.path.join(self.dirname, *args)

    def is_dir(self, basename: str) -> bool:
        """Checks if the given basename in the directory is a directory.

        Args:
            basename (str): The basename to check.

        Returns:
            bool: True if the basename is a directory, False otherwise.
        """
        return os.path.isdir(self.join(basename))

    def is_usage_dir(self, basename: str, usage: str) -> bool:
        """Checks if the given basename in the directory is a directory with 
        the given usage.

        Args:
            basename (str): The basename to check.
            usage (str): The usage to check.

        Returns:
            bool: True if the basename is a directory with the given usage, 
                False otherwise.
        """
        return (basename.startswith(usage) and self.is_dir(basename))

    def find_usage_basenames(self, usage: str) -> list[str]:
        """Finds all basenames in the directory with the given usage.

        Args:
            usage (str): The usage to check.

        Returns:
            list[str]: A list of basenames with the given usage in the 
                directory.
        """
        return [
            basename for basename in os.listdir(self.dirname)
            if self.is_usage_dir(basename, usage)
        ]

    def mkdir(self, basename: str) -> None:
        """Creates a directory with the given basename.

        Args:
            basename (str): The basename of the directory to create.
        """
        os.mkdir(self.join(basename))

    def get_usage_indices(self, usage: str) -> list[int]:
        """Finds all indices of directories with the given usage in the 
        directory.

        Args:
            usage (str): The usage to check.

        Returns:
            list[int]: A list of indices of directories with the given usage in 
                the directory.
        """
        return [
            int(basename.rsplit('_', maxsplit=1)[-1])
            for basename in self.find_usage_basenames(usage)
        ]

    def get_new_usage_idx(self, usage: str) -> str:
        """Finds the next available index for the directory with the given 
        usage.

        Args:
            usage (str): The usage to check.

        Returns:
            int: The next available index for the directory with the given 
                usage.
        """
        usage_indices = self.get_usage_indices(usage)
        idx = 0
        while idx in usage_indices:
            idx += 1
        return idx

    def get_new_usage_basename(self, usage: str) -> str:
        """Finds the next available basename for the directory with the given 
        usage.

        Args:
            usage (str): The usage to check.

        Returns:
            str: The next available basename for the directory with the given 
                usage.
        """
        return f'{usage}_{self.get_new_usage_idx(usage)}'

    def get_new_usage_dir(self, usage: str) -> str:
        """Finds the next available directory for the directory with the given

        Args:
            usage (str): The usage to check.

        Returns:
            str: The next available directory for the directory with the given 
                usage.
        """
        return self.join(self.get_new_usage_basename(usage))

    def __repr__(self) -> str:
        return self.dirname