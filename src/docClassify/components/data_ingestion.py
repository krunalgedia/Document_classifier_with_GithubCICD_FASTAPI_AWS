from docClassify.logger import logger
import os
import zipfile
from docClassify.logger import logger
from docClassify.entity import DataIngestionConfig
import subprocess


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if os.path.exists(self.config.local_data_file):
            os.remove(self.config.local_data_file)
        try:
            subprocess.run(['curl' ,'-L' ,'-o', self.config.local_data_file, self.config.source_URL], check=True)
            logger.info(f"File downloaded and saved as: {self.config.local_data_file}")
            #logger.info(f"{filename} download! with following info: \n{headers}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download file. Error: {e}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)








