import os
from box.exceptions import BoxValueError
import yaml
from docClassify.logger import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
from PIL import Image
import torch

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def create_bounding_box(bbox_data: List[Tuple[int, int]]) -> List[int]:
    xs = []
    ys = []

    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))

    return [left, top, right, bottom]


def scale_bounding_box(box: List[int], width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]

class DocumentClassificationDataset(Dataset):

    def __init__(self, image_paths: List[Path], processor):
        self.image_paths = image_paths
        self.processor = processor
        self.DOCUMENT_CLASSES = sorted(list(map(lambda p: p.name,image_paths[0].parent.parent.glob("*"))))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        json_path = image_path.with_suffix(".json")
        with json_path.open("r") as f:
            ocr_result = json.load(f)

            with Image.open(image_path).convert("RGB") as image:

                width, height = image.size
                width_scale = 1000 / width
                height_scale = 1000 / height

                words = []
                boxes = []
                for row in ocr_result:
                    boxes.append(scale_bounding_box(
                        row["bounding_box"],
                        width_scale,
                        height_scale
                    ))
                    words.append(row["word"])

                encoding = self.processor(
                    image,
                    words,
                    boxes=boxes,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

        label = self.DOCUMENT_CLASSES.index(image_path.parent.name)

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            bbox=encoding["bbox"].flatten(end_dim=1),
            pixel_values=encoding["pixel_values"].flatten(end_dim=1),
            labels=torch.tensor(label, dtype=torch.long)
        )

def compute_metrics(p):
    predictions, labels = p

    # Flatten the lists (assuming each element is a single-class prediction)
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]

    # Calculate metrics
    accuracy = accuracy_score(flat_labels, flat_predictions)
    precision = precision_score(flat_labels, flat_predictions, average='weighted')
    recall = recall_score(flat_labels, flat_predictions, average='weighted')
    f1 = f1_score(flat_labels, flat_predictions, average='weighted')

    return precision, recall, f1, accuracy