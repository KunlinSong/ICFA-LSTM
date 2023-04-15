import os
from contextlib import suppress

import torch

import icfalstm.nn as mynn
import icfalstm.utils as myutils
from icfalstm.types import *


class Trainer:

    def __init__(self) -> None:
        self.root_dirname = os.getcwd()
        self.root_dir = myutils.Directory(self.root_dirname)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64
        self._init_config()
        self._init_dataset()
        self._init_dataloader()
        self._init_logger()
        self._init_model()
        self._init_recorder()
        self.logger.add_graph(self.model)

    def _init_config(self) -> None:
        config_saver_path = self.root_dir.join('custom',
                                               'config_saver_settings.json')
        config_path = self.root_dir.join('custom', 'config.json')
        config_saver = myutils.ConfigSaver(config_saver_path)
        self.config = myutils.Config(config_path, config_saver)

    def _init_dataset(self) -> None:
        data_dict = self._get_data_dict(self.config)
        self.data_set = self._get_dataset(data_dict)

    def _get_dataset(self, data_dict) -> None:
        npz_data_dirname = self.root_dir.join('generated', 'npz_data')
        npz_data_dir = myutils.Directory(npz_data_dirname)
        npz_data_foldernames = npz_data_dir.find_usage_basenames('npz_data')
        for npz_data_foldername in npz_data_foldernames:
            npz_data_config_path = npz_data_dir.join(npz_data_foldername,
                                                     'config.json')
            npz_data_config = myutils.Config(npz_data_config_path,
                                             self.config.config_saver)
            if self.config.is_equal(npz_data_config, 'npz_data'):
                npz_data_dirname = npz_data_dir.join(npz_data_foldername)
                self.dataset = myutils.Dataset(data_dict, npz_data_dirname,
                                               'npz')
                return
        csv_data_dirname = self.root_dir.join('csv_data')
        return myutils.Dataset(data_dict, csv_data_dirname, 'csv')

    def _get_data_dict(self, config) -> myutils.DataDict:
        csv_data_dirname = self.root_dir.join('csv_data')
        data_dict_dirname = self.root_dir.join('generated', 'data_dict')
        if not os.path.exists(data_dict_dirname):
            os.makedirs(data_dict_dirname)
        try:
            data_dict = myutils.DataDict.from_saved(data_dict_dirname, config)
        except FileNotFoundError:
            data_dict = myutils.DataDict.from_data(csv_data_dirname, config)
            data_dict.save(data_dict_dirname)
        return data_dict

    def _init_dataloader(self):
        self.dataset.switch_to('train')
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            num_workers=os.cpu_count(),
            shuffle=True)
        self.dataset.switch_to('val')
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            num_workers=os.cpu_count(),
            shuffle=False)
        self.dataset.switch_to('test')
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=os.cpu_count(),
            shuffle=False)

    def _init_logger(self):
        model_dirname = self.root_dir.join('generated', 'model')
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)
        model_dir = myutils.Directory(model_dirname)
        model_foldernames = model_dir.find_usage_basenames('model')
        for model_foldername in model_foldernames:
            model_config_path = model_dir.join(model_foldername, 'config.json')
            model_config = myutils.Config(model_config_path,
                                          self.config.config_saver)
            if self.config.is_equal(model_config, 'model'):
                model_dirname = model_dir.join(model_foldername)
                self.logger = myutils.Logger(model_dirname, self.config)
                return
        model_dirname = model_dir.get_new_usage_dir('model')
        os.makedirs(model_dirname)
        self.config.save(model_dirname, 'model')
        self.logger = myutils.Logger(model_dirname, self.config)

    def _init_model(self):
        self.model = mynn.RNNBase(mode=self.config['mode'],
                                  map_units=len(self.config['cities']),
                                  in_features=len(self.config['attributes']),
                                  hidden_units=self.config['hidden_units'],
                                  out_features=len(self.config['targets']),
                                  dtype=self.dtype,
                                  device=self.device)
        with suppress(FileNotFoundError):
            model_state_dict = self.logger.get_state_dict(
                'latest', self.device)
            self.model.load_state_dict(model_state_dict)
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer, T_0=16)

    def _init_recorder(self):
        self.time_recorder = myutils.TimeRecorder()
        self.train_loss_recorder = myutils.LossRecorder(
            self.config['max_epoch'])
        self.val_loss_recorder = myutils.LossRecorder(self.config['max_epoch'])

    def load_best_state_dict(self):
        with suppress(FileNotFoundError):
            model_state_dict = self.logger.get_state_dict('best', self.device)
            self.model.load_state_dict(model_state_dict)


if __name__ == '__main__':
    print('Preparing ...')
    trainer = Trainer()
    for epoch in range(trainer.logger.start_epoch):
        if trainer.logger.early_stopping(epoch):
            break
        trainer.scheduler.step()
    for epoch in range(trainer.logger.start_epoch,
                       trainer.config['max_epoch'] + 1):
        if trainer.logger.early_stopping(epoch):
            break
        myutils.print_separator()
        print(f'Epoch: {epoch}/{trainer.config["max_epoch"]}')

        print('Train')
        trainer.time_recorder.update_time('start')
        trainer.time_recorder.print_time('start')
        trainer.dataset.switch_to('train')
        trainer.train_loss_recorder.new_epoch(epoch, len(trainer.train_loader))
        for i, (cpu_inputs, cpu_targets) in enumerate(trainer.train_loader,
                                                      start=1):
            inputs = cpu_inputs.to(trainer.device)
            targets = cpu_targets.to(trainer.device)
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            loss = trainer.loss(outputs, targets)
            loss.backward()
            trainer.optimizer.step()
            trainer.train_loss_recorder.update(i, loss.item())
            trainer.train_loss_recorder.print_batch_loss()
        print()
        trainer.train_loss_recorder.print_epoch_loss()
        trainer.time_recorder.update_time('end')
        trainer.time_recorder.print_time('end')
        trainer.time_recorder.print_spend()

        print('Validation')
        trainer.time_recorder.update_time('start')
        trainer.time_recorder.print_time('start')
        trainer.dataset.switch_to('val')
        trainer.val_loss_recorder.new_epoch(epoch, len(trainer.val_loader))
        for i, (cpu_inputs, cpu_targets) in enumerate(trainer.val_loader,
                                                      start=1):
            with torch.no_grad():
                inputs = cpu_inputs.to(trainer.device)
                targets = cpu_targets.to(trainer.device)
                outputs = trainer.model(inputs)
                loss = trainer.loss(outputs, targets)
                trainer.val_loss_recorder.update(i, loss.item())
                trainer.val_loss_recorder.print_batch_loss()
        print()
        trainer.val_loss_recorder.print_epoch_loss()
        trainer.time_recorder.update_time('end')
        trainer.time_recorder.print_time('end')
        trainer.time_recorder.print_spend()

        print('Saving logs ...')
        trainer.logger.add_loss('train',
                                trainer.train_loss_recorder.get_epoch_loss(),
                                epoch)
        trainer.logger.add_loss('val',
                                trainer.val_loss_recorder.get_epoch_loss(),
                                epoch)
        trainer.logger.save_state_dict(trainer.model.state_dict(), 'latest')
        if trainer.logger.is_best(epoch,
                                  trainer.val_loss_recorder.get_epoch_loss()):
            trainer.logger.save_state_dict(trainer.model.state_dict(), 'best')
        trainer.scheduler.step()

    myutils.print_separator()
    print('Testing ...')
    trainer.load_best_state_dict()
    trainer.time_recorder.update_time('start')
    trainer.time_recorder.print_time('start')
    true_values = []
    predicted_values = []
    trainer.dataset.switch_to('test')
    test_len = len(trainer.test_loader)
    for i, (inputs, targets) in enumerate(trainer.test_loader, start=1):
        with torch.no_grad():
            outputs = trainer.model(inputs)
            true_values.append(torch.transpose(targets, 0, 1)[-1])
            predicted_values.append(torch.transpose(outputs, 0, 1)[-1])
        print(f'\rTest finished: {i}/{test_len}', end='')
    true_values = torch.cat(true_values, dim=0)
    predicted_values = torch.cat(predicted_values, dim=0)
    print('\nTest all finished.')
    print('\nAdd info to logs ...')
    trainer.logger.save_predicted_true(predicted_values, true_values)
    trainer.logger.add_test_info()
    trainer.logger.plot_assoc_mat()
    trainer.time_recorder.update_time('end')
    trainer.time_recorder.print_time('end')
    trainer.time_recorder.print_spend()
    myutils.print_separator()