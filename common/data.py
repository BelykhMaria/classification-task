import numpy as np
from loguru import logger
import torch
from torch.utils.data import Dataset

from common.utils import get_paths_to_npy_files, read_npy_file
from config.config import settings


class MelClassDataset(Dataset):
    '''
    Mel-spectrogram - dataset for classification.
    '''

    def __init__(self, root_dir: str, max_length: int=1500, clean_folder_name: str='clean'):
        self.root_dir = root_dir
        self.paths_to_files = get_paths_to_npy_files(start_folder=root_dir)
        self.max_length = max_length
        self.clean_folder_name = clean_folder_name


    def __len__(self) -> int:
        return len(self.paths_to_files)


    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mel = read_npy_file(self.paths_to_files[idx])
        mel = mel.T
        mel = mel.astype('float32')

        mel = np.pad(mel, pad_width=((0, 0), (0, self.max_length - mel.shape[1])), mode='constant', constant_values=1e-6)

        if self.paths_to_files[idx].find(self.clean_folder_name) != -1:
            label = 0
        else:
            label = 1
        sample = {settings.TRAIN.INPUT_LABEL : mel, settings.TRAIN.OUTPUT_LABEL : label}

        return sample
