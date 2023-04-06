import datetime
import os
from typing import Literal

import torch

import icfalstm.model as model
import icfalstm.util as util


def mean_loss(epoch_loss: list[float]) -> float:
    """Calculates the mean loss.

    Args:
        epoch_loss (list[float]): The epoch loss.

    Returns:
        float: The mean loss.
    """
    return sum(epoch_loss) / len(epoch_loss)


class Training:
    """A class for training the model.

    Attributes:
        device (torch.device): The device.
        config (util.Config): The configuration object.
        setting (util.Setting): The setting object.
        time_recorder (util.TimeRecorder): The time recorder object.
        dataset (util.Dataset): The dataset object.
        dataloader (torch.utils.data.DataLoader): The dataloader object.
        validation_dataloader (torch.utils.data.DataLoader): The
            validation dataloader object.
        using_model_dirname (str): The directory name of the using model.
        logger (util.Logger): The logger object.
        model (model.Model): The model object.
        loss (torch.nn.modules.loss.L1Loss): The loss object.
        optimizer (torch.optim.adam.Adam): The optimizer object.
    """
    BASIC_FEATURES = ('csv_data', 'custom')
    GENERATE_FEATURES = ('npz_data', 'model')

    def __init__(self, root_dirname: str) -> None:
        """Initializes a Training object.

        Args:
            root_dirname (str): The root directory name.
        """
        for feature in Training.BASIC_FEATURES:
            setattr(self, f'{feature}_dirname',
                    os.path.join(root_dirname, feature))
        for feature in Training.GENERATE_FEATURES:
            setattr(self, f'{feature}_dirname',
                    os.path.join(root_dirname, 'generate', feature))
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = util.Config(self._get_custom_path('config.txt'))
        self.setting = util.Setting(self._get_custom_path('setting.txt'))
        self.time_recorder = util.TimeRecorder()
        self.dataset = self._get_dataset()
        self.dataloader = self._get_dataloader()
        self.validation_dataloader = self._get_dataloader('validation')
        self.using_model_dirname = self._get_using_model_dirname()
        self.logger = util.Logger(self.using_model_dirname, self.config)
        self.model = self._get_model()
        self.loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)
        self.logger.add_graph(self.model)

    def _get_custom_path(
            self, filename: Literal['config.txt', 'setting.txt']) -> str:
        """Gets the path of the custom file.

        Args:
            filename (['config.txt', 'setting.txt']): The filename.

        Returns:
            str: The path of the custom file.
        """
        return os.path.join(self.custom_dirname, filename)

    def _get_using_model_dirname(self) -> str:
        """Gets the directory name of the using model.

        Returns:
            str: The directory name of the using model.
        """
        directory = util.Directory(self.model_dirname)
        all_model_foldername = directory.get_exist_folder_for_usage('model')
        for model_foldername in all_model_foldername:
            model_dirname = os.path.join(self.model_dirname, model_foldername)
            model_config = util.Config(
                os.path.join(model_dirname, 'config.txt'))
            if self.config.is_equal_to_for_usage(model_config, 'model'):
                return model_dirname
        model_foldername = directory.get_new_foldername('model')
        directory.mkdir(model_foldername)
        model_dirname = os.path.join(self.model_dirname, model_foldername)
        datadict = self.dataset.datadict
        self.config.save_for_usage(
            model_dirname,
            self.setting,
            'model',
            train_length=datadict.train_length,
            validation_length=datadict.validation_length,
            test_length=datadict.test_length)
        return model_dirname

    def _get_dataset(self) -> util.Dataset:
        """Gets the dataset.

        Returns:
            util.Dataset: The dataset.
        """
        npz_data_directory = util.Directory(self.npz_data_dirname)
        all_data_foldername = npz_data_directory.get_exist_folder_for_usage(
            'data')
        for data_foldername in all_data_foldername:
            data_dirname = os.path.join(self.npz_data_dirname, data_foldername)
            data_config = util.Config(os.path.join(data_dirname, 'config.txt'))
            if self.config.is_equal_to_for_usage(data_config, self.setting,
                                                 'data'):
                datadict = util.DataDict(data_dirname,
                                         self.config,
                                         file_type='npz')
                return util.Dataset(datadict, self.config)
        datadict = util.DataDict(self.csv_data_dirname,
                                 self.config,
                                 file_type='csv')
        return util.Dataset(datadict, self.config)

    def _get_model(self) -> torch.nn.Module:
        """Gets the model.

        Returns:
            torch.nn.Module: The model.
        """
        train_model = model.RNNBase(mode=self.config.mode,
                                    input_size=(len(self.config.cities),
                                                len(self.config.attributes)),
                                    hidden_size=self.config.hidden_units,
                                    num_outputs=len(self.config.targets),
                                    device=self.device,
                                    batch_first=True)
        if os.path.exists(self.logger.latest_model):
            train_model.load_state_dict(
                self.logger.load_state_dict('latest', self.device))

    def _get_dataloader(self) -> torch.utils.data.DataLoader:
        """Gets the dataloader.

        Returns:
            torch.utils.data.DataLoader: The dataloader.
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=16,
            shuffle=(self.dataset.state == 'train'))


if __name__ == '__main__':
    root_dirname = os.getcwd()
    training = Training(root_dirname)
    for epoch in range(training.config.max_epoch):
        if epoch < training.logger.start_epoch:
            continue
        if training.logger.early_stopping(epoch):
            break

        util.print_divide_line()
        print(f'Epoch: {epoch} / {training.config.max_epoch}')

        print('Train')
        training.time_recorder.get_time('start')
        training.time_recorder.print_start()
        training.dataset.switch_to('train')
        train_epoch_loss = []
        train_epoch_last_prediction_loss = []
        for batch_idx, (input_data,
                        target_data) in enumerate(training.dataloader,
                                                  start=1):
            training.optimizer.zero_grad()
            output_data = training.model(input_data)
            loss = training.loss(output_data, target_data)
            loss.backward()
            training.optimizer.step()
            train_epoch_loss.append(loss.item())
            last_prediction_loss = training.loss(
                output_data.permute(1, 0, 2, 3)[-1],
                target_data.permute(1, 0, 2, 3)[-1])
            train_epoch_last_prediction_loss.append(
                last_prediction_loss.item())
            util.print_batch_loss(last_prediction_loss.item(), batch_idx,
                                  len(training.dataloader))
        print('\n')
        train_epoch_loss = mean_loss(train_epoch_loss)
        train_epoch_last_prediction_loss = mean_loss(
            train_epoch_last_prediction_loss)
        util.print_epoch_loss(train_epoch_loss)
        training.time_recorder.get_time('end')
        training.time_recorder.print_end()
        training.time_recorder.print_spend()

        print('Validation')
        training.time_recorder.get_time('start')
        training.time_recorder.print_start()
        training.dataset.switch_to('validation')
        validation_epoch_loss = []
        validation_epoch_last_prediction_loss = []
        for batch_idx, (input_data,
                        target_data) in enumerate(training.dataloader,
                                                  start=1):
            output_data = training.model(input_data)
            loss = training.loss(output_data, target_data)
            validation_epoch_loss.append(loss.item())
            last_prediction_loss = training.loss(
                output_data.permute(1, 0, 2, 3)[-1],
                target_data.permute(1, 0, 2, 3)[-1])
            validation_epoch_last_prediction_loss.append(
                last_prediction_loss.item())
            util.print_batch_loss(last_prediction_loss.item(), batch_idx,
                                  len(training.dataloader))
        print('\n')
        validation_epoch_loss = mean_loss(validation_epoch_loss)
        validation_epoch_last_prediction_loss = mean_loss(
            validation_epoch_last_prediction_loss)
        util.print_epoch_loss(validation_epoch_loss)
        training.time_recorder.get_time('end')
        training.time_recorder.print_end()
        training.time_recorder.print_spend()

        training.logger.add_loss('train', train_epoch_loss, epoch)
        training.logger.add_loss('train_last_prediction',
                                 train_epoch_last_prediction_loss, epoch)
        training.logger.add_loss('validation', validation_epoch_loss, epoch)
        training.logger.add_loss('validation_last_prediction',
                                 validation_epoch_last_prediction_loss, epoch)

        training.logger.save_state_dict(training.model, 'latest')
        if training.logger.is_best(epoch, validation_epoch_loss, 'best'):
            training.logger.save_state_dict(training.model, 'best')
        if training.logger.is_best(epoch,
                                   validation_epoch_last_prediction_loss,
                                   'last_prediction_best'):
            training.logger.save_state_dict(training.model,
                                            'last_prediction_best')
