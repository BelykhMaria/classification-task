import torch
import random

from train.train import TrainClassModel
from common.model import GruNet
from common.data import MelClassDataset
from config.config import settings


if __name__ == "__main__":
    random.seed(10)
    model = GruNet(hidden_dim=30, input_dim=1500, output_dim=1, bidirectional=True)
    train_dataset = MelClassDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.TRAIN_FOLDER_NAME)
    valid_dataset = MelClassDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.VALID_FOLDER_NAME)
    optimizer = torch.optim.Adam
    loss = torch.nn.BCEWithLogitsLoss()
    train_class_model = TrainClassModel(loss=loss, optimizer=optimizer, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    train_class_model()
