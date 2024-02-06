from tqdm import tqdm
import torch
import pandas as pd
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from docClassify.logger import logger
from docClassify.entity import DataTrainingValidationConfig
from pathlib import Path
from docClassify.utils.common import DocumentClassificationDataset
from docClassify.constants import *
from docClassify.utils.common import read_yaml, create_directories, compute_metrics, scale_bounding_box, create_bounding_box


class TrainAndValidate:
    def __init__(self, config: DataTrainingValidationConfig):
        self.feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = LayoutLMv3Processor(self.feature_extractor, self.tokenizer)
        self.config = config
        self.DOCUMENT_CLASSES = sorted(list(map(lambda p: p.name, Path(self.config.unzip_dir).glob("*"))))

    def get_train_test_path(self):
        # Convert PosixPath objects to strings
        image_paths = sorted(list(Path(self.config.unzip_dir).glob("*/*.png")))
        image_paths_str = [str(path) for path in image_paths]

        # Define labels based on whether the paths contain specific strings
        income_labels = ["income" in path for path in image_paths_str]
        balance_labels = ["balance" in path for path in image_paths_str]
        cashflow_labels = ["cashflow" in path for path in image_paths_str]

        # Use any one of the labels as the target for stratified split
        # Here, I'm using income_labels, but you can choose based on your requirements
        train_images_str, test_images_str = train_test_split(image_paths_str, test_size=0.2, stratify=income_labels,
                                                             random_state=42)

        # Convert back to PosixPath objects
        train_images = [Path(path) for path in train_images_str]
        test_images = [Path(path) for path in test_images_str]

        return train_images, test_images

    def train(self, train_images, test_images):
        train_dataset = DocumentClassificationDataset(train_images, self.processor)
        valid_dataset = DocumentClassificationDataset(test_images, self.processor)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            # num_workers=10
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            # num_workers=10
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        n_classes = len(self.DOCUMENT_CLASSES)

        model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )
        model.to(device)

        # load seqeval metric
        # metric = evaluate.load("seqeval")
        model.config.id2label = {k: v for k, v in enumerate(self.DOCUMENT_CLASSES)}
        model.config.label2id = {v: k for k, v in enumerate(self.DOCUMENT_CLASSES)}
        # labels of the model
        ner_labels = list(model.config.id2label.values())

        num_epochs = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

        # Initialize an empty DataFrame to store the metrics
        columns = ["Epoch", "Training Loss", "Validation Loss", "Precision", "Recall", "F1", "Accuracy"]
        df_metrics = pd.DataFrame(columns=columns)

        # Early stopping parameters
        patience = 3  # Number of epochs to wait for improvement
        best_validation_loss = float('inf')
        current_patience = 0

        for epoch in range(num_epochs):
            print("Epoch:", epoch)

            # Training
            model.train()
            training_loss = 0.0
            num = 0
            for batch in tqdm(train_dataloader):
                labels = torch.Tensor(batch["labels"]).unsqueeze_(0).long().to(device)
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=torch.tensor(batch["attention_mask"]).to(device),
                    bbox=torch.tensor(batch["bbox"]).to(device),
                    pixel_values=torch.tensor(batch["pixel_values"]).to(device),
                    labels=batch["labels"].to(device)
                )
                loss = outputs.loss
                training_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                num += 1

            print("Training Loss:", training_loss / num)

            # Validation
            model.eval()
            preds = []
            labs = []
            validation_loss = 0.0
            num = 0
            for batch in tqdm(valid_dataloader):
                labels = torch.Tensor(batch["labels"]).to(device)
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=torch.tensor(batch["attention_mask"]).to(device),
                    bbox=torch.tensor(batch["bbox"]).to(device),
                    pixel_values=torch.tensor(batch["pixel_values"]).to(device),
                    labels=labels
                )
                loss = outputs.loss
                preds_idx = outputs.logits.argmax(axis=1)
                labs.append(labels.tolist())
                preds.append(preds_idx.tolist())
                validation_loss += loss.item()
                num += 1

            print("Validation Loss:", validation_loss / num)
            print(preds)
            print(labs)

            overall_precision, overall_recall, overall_f1, overall_accuracy = compute_metrics([preds, labs])
            print("Overall Precision:", overall_precision)
            print("Overall Recall:", overall_recall)

            # Store metrics in the DataFrame
            metrics_data = {
                "Epoch": epoch,
                "Training Loss": training_loss,
                "Validation Loss": validation_loss,
                "Precision": overall_precision,
                "Recall": overall_recall,
                "F1": overall_f1,
                "Accuracy": overall_accuracy
            }
            # df_metrics = df_metrics.append(metrics_data, ignore_index=True)
            df_metrics.loc[len(df_metrics)] = metrics_data

            # Early stopping check
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f"Early stopping! No improvement in validation loss for {patience} consecutive epochs.")
                    break

        # Save the DataFrame to a CSV file or do any further analysis
        df_metrics.to_csv("metrics.csv", index=False)
        print(df_metrics)

        return df_metrics

        # Convert DataFrame to markdown
        # markdown_table = df_metrics.to_markdown()

        # Print the markdown table
        # print(markdown_table)

