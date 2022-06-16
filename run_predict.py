from predict.predict import PredictClassModel
from common.data import MelClassDataset
from config.config import settings

if __name__ == "__main__":
    test_dataset = MelClassDataset(root_dir=settings.DATA.DATA_PATH + '\\' + settings.DATA.TEST_FOLDER_NAME)
    y_predict = PredictClassModel(test_dataset = test_dataset)()
