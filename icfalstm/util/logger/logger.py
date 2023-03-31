import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, dirname) -> None:
        self.latest_model = os.path.join(dirname, 'latest_model.pth')
        self.best_model = os.path.join(dirname, 'best_model.pth')
        self.training_loss = os.path.join(dirname, 'training_loss.csv')
        self.training_last_prediction_loss = os.path.join(dirname, 'training_last_prediction_loss.csv')
        self.validation_loss = os.path.join(dirname, 'validation_loss.csv')
        self.validation_last_prediction_loss = os.path.join(dirname, 'validation_last_prediction_loss.csv')
        self.testing_target_prediction = os.path.join(dirname, 'testing_target_prediction.csv')
        self.testing_info = os.path.join(dirname, 'testing_info.csv')
        self.writer = SummaryWriter(os.path.join(dirname, 'logs'))
