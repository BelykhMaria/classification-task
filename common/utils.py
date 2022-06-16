import os
import numpy as np
from loguru import logger
import torch

from config.config import settings


def read_npy_file(path_to_file: str) -> np.ndarray:
    '''
    This function reads npy file.

    Parameters:
    -----------
        path_to_file: str
            Path to file with name of the file.

    Returns:
    -------
        audio_file: np.ndarray
            Array representation of audio file.
    '''

    return np.load(path_to_file, allow_pickle=True)


def get_paths_to_npy_files(start_folder: str, ext: str = '.npy') -> list:
    '''
    This function returns list of all paths to files in start folder that have specified extension.

    Parameters:
    ----------

    start_folder: str
        The folder to start search from.
    ext: str (default = '.npy')
        The extension of files to search.

    Returns:
    -------

        filenames: list
            List of paths to all files in subfolders of start_folder with ext extension.
    '''

    logger.info(f'getting all paths to {ext} format in {start_folder}')
    return [os.path.join(root, name) for root, _, files in os.walk(start_folder) for name in files if name.endswith((ext))]


def binary_acc(y_pred, y_test) -> torch.Tensor:
    '''
    This function calcualtes accuarcy score for a given prediction y_pred.
    '''

    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def weights_init_uniform(m):
    '''
    This function initialize model with uniformly distributed weights.
    '''

    logger.info('initializing model weights')
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
