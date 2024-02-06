import easyocr
from tqdm import tqdm
import json
import os
from docClassify.logger import logger
from docClassify.entity import DataPreparationConfig
from pathlib import Path
from docClassify.constants import *
from docClassify.utils.common import read_yaml, create_directories, create_bounding_box


class DataPreparation():
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        # print(Path(self.config.root_dir).glob("*/*.png"))
        self.image_paths = sorted(list(Path(self.config.unzip_dir).glob("*/*.png")))
        # print(Path(self.config.root_dir).glob("*/*.png"))

    def prepare_all_files(self) -> bool:
        preparation_status = False
        try:
            reader = easyocr.Reader(['en'])
            for image_path in tqdm(self.image_paths):
                print(image_path)
                ocr_result = reader.readtext(str(image_path))

                ocr_page = []
                for bbox, word, confidence in ocr_result:
                    ocr_page.append({
                        "word": word, "bounding_box": create_bounding_box(bbox)
                    })

                with image_path.with_suffix(".json").open("w") as f:
                    json.dump(ocr_page, f)

            preparation_status = True
            return preparation_status

        except Exception as e:
            raise e