import os
from glob import glob
import icfalstm.util as util


def get_config_path(model_dirname: str) -> str:
    """Get the path to the config file.

    Args:
        model_dirname (str): The directory name of the config file.

    Returns:
        str: The path to the config file.
    """
    return os.path.join(model_dirname, 'config.txt')

class RelPath:
    """A class for getting the relative path to a file or directory.

    Attributes:
        start (str): The start directory.
    """
    def __init__(self, start: str) -> None:
        """Initializes a RelPath object.

        Args:
            start (str): The start directory.
        """
        self.start = start
    
    def __call__(self, *args: str) -> str:
        """Gets the relative path to a file or directory.

        Returns:
            str: The relative path to a file or directory.
        """
        return os.path.relpath(os.path.join(*args), start=self.start)


if __name__ == '__main__':
    root_dirname = os.getcwd()
    generate_dirname = os.path.join(root_dirname, 'generate')
    base_model_dirname = os.path.join(generate_dirname, 'model')
    model_dir = util.Directory(base_model_dirname)
    model_folder_list = model_dir.get_exist_folder_for_usage('model')
    model_dict = {
        model_folder: os.path.join(base_model_dirname, model_folder)
        for model_folder in model_folder_list
    }

    model_config_summary_path = os.path.join(generate_dirname,
                                             'model_config_summary.md')
    with open(model_config_summary_path, 'w') as f:
        model_config_summary_dirname = os.path.dirname(
            model_config_summary_path)
        relpath = RelPath(model_config_summary_dirname)
        f.writelines((
            f'{model_folder}: '
            f'(dir)[{relpath(model_dirname)}] '
            f'(config)[{relpath(model_dirname, "config.txt")}]'
            for model_folder, model_dirname in model_dict.items()))
