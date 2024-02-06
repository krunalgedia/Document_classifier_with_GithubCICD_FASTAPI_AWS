from docClassify.config.configuration import ConfigurationManager
from docClassify.components.data_preparation import DataPreparation
from docClassify.logger import logger

class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.prepare_all_files()