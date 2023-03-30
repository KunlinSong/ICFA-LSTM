import os
from glob import glob
import icfalstm.util as util


if __name__ == '__main__':
    root_path = os.getcwd()
    csv_path = os.path.join(root_path, 'csv_data')
    npz_path = os.path.join(root_path, 'generate', 'npz_data')
    
    custom_path = os.path.join(root_path, 'custom')
    config_path = os.path.join(custom_path, 'config.txt')
    setting_path = os.path.join(custom_path, 'setting.txt')

    config = util.Config(config_path)
    setting = util.Setting(setting_path)

    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    csv_data_list = glob(os.path.join(csv_path, '*.csv'))
    for csv_data in csv_data_list:
        data = util.CSVData(csv_data, config)
        data.to_npz(npz_path)
    config.save_for_usage(npz_path, setting, 'data')