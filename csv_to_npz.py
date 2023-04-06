import os
from glob import glob
import icfalstm.util as util


if __name__ == '__main__':
    root_dirname = os.getcwd()
    csv_dirname = os.path.join(root_dirname, 'csv_data')
    npz_dirname = os.path.join(root_dirname, 'generate', 'npz_data')
    if not os.path.exists(npz_dirname):
        os.makedirs(npz_dirname)
    npz_dir = util.Directory(npz_dirname)
    data_to_foldername = npz_dir.get_new_foldername('data')
    
    custom_dirname = os.path.join(root_dirname, 'custom')
    config_path = os.path.join(custom_dirname, 'config.txt')
    setting_path = os.path.join(custom_dirname, 'setting.txt')

    config = util.Config(config_path)
    setting = util.Setting(setting_path)

    npz_dir.mkdir(data_to_foldername)
    data_to_dirname = os.path.join(npz_dirname, data_to_foldername)
    csv_data_list = glob(os.path.join(csv_dirname, '*.csv'))
    for idx, csv_data in enumerate(csv_data_list, start=1):
        data = util.CSVData(csv_data, config)
        data.to_npz(data_to_dirname)
        print(f'\rFinished: {idx / len(csv_data_list) * 100:.2f}% '
              f'{idx} / {len(csv_data_list)}', end='')
    config.save_for_usage(data_to_dirname, setting, 'data')
    print('\n')