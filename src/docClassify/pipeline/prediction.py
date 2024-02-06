import os
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import easyocr
from docClassify.utils.common import scale_bounding_box, create_bounding_box
import torch
from PIL import Image


class Predictor:
    def __init__(self, image_path):
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "KgModel/IncomeStatement_Cashflow_BalanceStatement__Classifier_LayoutLMv3")
        self.feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = LayoutLMv3Processor(self.feature_extractor, self.tokenizer)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.reader = easyocr.Reader(['en'])
        self.image_path = image_path

    def predict(self):

        ocr_result = self.reader.readtext(str(self.image_path))
        ocr_page = []
        for bbox, word, confidence in ocr_result:
            ocr_page.append({
                "word": word, "bounding_box": create_bounding_box(bbox)
            })

        # with Image.open(image_path).convert("RGB") as image:

        image = Image.open(self.image_path).convert("RGB")
        width, height = image.size
        width_scale = 1000 / width
        height_scale = 1000 / height
        words = []
        boxes = []
        for row in ocr_page:
            boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
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

        outputs = self.model(
            input_ids=encoding["input_ids"].to(self.device),
            attention_mask=torch.tensor(encoding["attention_mask"]).to(self.device),
            bbox=torch.tensor(encoding["bbox"]).to(self.device),
            pixel_values=torch.tensor(encoding["pixel_values"]).to(self.device),
        )

        preds_idx = outputs.logits.argmax(axis=1)

        return self.model.config.id2label[int(preds_idx[0])]

