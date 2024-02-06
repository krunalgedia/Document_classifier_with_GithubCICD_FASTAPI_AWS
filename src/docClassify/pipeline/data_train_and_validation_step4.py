from docClassify.config.configuration import ConfigurationManager
from docClassify.components.data_train_and_validation import TrainAndValidate
from docClassify.logger import logger

class DataTrainValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_training_validation_config = config.get_data_training_validation_config()
        train_and_validate = TrainAndValidate(data_training_validation_config)
        train_images, test_images = train_and_validate.get_train_test_path()
        train_and_validate.train(train_images, test_images)