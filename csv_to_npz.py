import os
import sys
from glob import glob

import icfalstm.utils as myutils


if __name__ == '__main__':
    # Get the needed directories and paths.
    root_dirname = os.getcwd()
    root_dir = myutils.Directory(root_dirname)
    csv_dirname = root_dir.join('csv_data')
    npz_dirname = root_dir.join('generated', 'npz_data')
    if not os.path.exists(npz_dirname):
        os.makedirs(npz_dirname)
    npz_dir = myutils.Directory(npz_dirname)

    config_path = root_dir.join('custom', 'config.json')
    config_saver_path = root_dir.join('custom', 'config_saving_settings.json')
    config_saver = myutils.ConfigSaver(config_saver_path)
    config = myutils.Config(config_path, config_saver)

    # Check if the npz data already exists.
    npz_data_folders = npz_dir.find_usage_basenames('npz_data')
    for npz_data_folder in npz_data_folders:
        npz_data_config_path = npz_dir.join(npz_data_folder, 'config.json')
        npz_data_config = myutils.Config(npz_data_config_path, config_saver)
        if config.is_equal(npz_data_config, 'npz_data'):
            print(f'NPZData {npz_data_folder} already exists.')
            sys.exit()
    
    # Convert the csv data to npz data.
    data_to_dirname = npz_dir.get_new_usage_dir('npz_data')
    os.makedirs(data_to_dirname)
    csv_data_lst = glob(os.path.join(csv_dirname, '*.csv'))
    num_csv_data = len(csv_data_lst)
    for idx, csv_data_path in enumerate(csv_data_lst, start=1):
        csv_data = myutils.CSVData(csv_data_path, config)
        csv_data.to_npz(data_to_dirname)
        print(f'\rFinished: {idx / num_csv_data * 100:.4g}% '
              f'[{idx} / {num_csv_data}]', end='')
    config.save(data_to_dirname, 'npz_data')
    print('\nAll Finished.')

