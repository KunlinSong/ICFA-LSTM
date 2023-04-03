import os
from glob import glob
import icfalstm.util as util


if __name__ == '__main__':
    root_dirname = os.getcwd()
    csv_dirname = os.path.join(root_dirname, 'csv_data')
    npz_dirname = os.path.join(root_dirname, 'generate', 'npz_data')
    if not os.path.exists(npz_dirname):
        os.makedirs(npz_dirname)
    npz_dir = util.Direction(npz_dirname)
    data_to_dirname = npz_dir.get_new_foldername('data')
    
    custom_dirname = os.path.join(root_dirname, 'custom')
    config_path = os.path.join(custom_dirname, 'config.txt')
    setting_path = os.path.join(custom_dirname, 'setting.txt')

    config = util.Config(config_path)
    setting = util.Setting(setting_path)

    npz_dir.mkdir(data_to_dirname)
    csv_data_list = glob(os.path.join(csv_dirname, '*.csv'))
    for csv_data in csv_data_list:
        data = util.CSVData(csv_data, config)
        data.to_npz(data_to_dirname)
    config.save_for_usage(data_to_dirname, setting, 'data')