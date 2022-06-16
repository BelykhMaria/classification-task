import torch
from tqdm import tqdm
import numpy as np
import copy

from torch.utils.data import DataLoader
from config.config import settings
from common.utils import binary_acc
from common.model import GruNet

class PredictClassModel():
    def __init__(self, test_dataset,
                model_path: str = settings.DATA.CLASS_MODEL_PATH,
                batch_size: int = settings.TRAIN.BATCH_SIZE):

        model = GruNet(hidden_dim=30, input_dim=1500, output_dim=1, bidirectional=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(copy.deepcopy(torch.load(model_path,device)))
        model.to(device)
        self.model = model
        self.device = device
        self.model_path = model_path 
        self.test_dataset = test_dataset
        self.batch_size = batch_size


    def predict(self) -> np.array:
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        model = self.model
        model.eval()

        test_acc = 0
        y_predict = []
        for batch in tqdm(test_loader):
            X_batch = batch[settings.TRAIN.INPUT_LABEL]
            y_batch = batch[settings.TRAIN.OUTPUT_LABEL]
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            with torch.no_grad():
                y_pred = model(X_batch)
                y_pred = y_pred.flatten()
                y_pred = y_pred.double()
                y_batch = y_batch.double()

                test_acc += binary_acc(y_pred, y_batch)
                y_predict.append(y_pred)

        print(f'Test accuracy: {test_acc/len(test_loader)}')
        return np.array(y_predict)


    def __call__(self):
        return self.predict()
