#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "microsoft/table-transformer-structure-recognition"

TRAIN_ANN = "/home/meeran/thesis_data/tatr_finetune_100ann/annotations/train.json"
VAL_ANN = "/home/meeran/thesis_data/tatr_finetune_100ann/annotations/val.json"

TRAIN_IMG_DIR = "/home/meeran/thesis_data/dataset_100ann_5class/train/images"
VAL_IMG_DIR = "/home/meeran/thesis_data/dataset_100ann_5class/val/images"

OUT_DIR = "/home/meeran/thesis_data/tatr_finetuned_100ann"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.1
FP16 = False
BF16 = False


class TATRFineTuneDataset(Dataset):
    def __init__(self, ann_path, img_dir, processor):
        with open(ann_path) as f:
            self.coco = json.load(f)

        self.img_dir = img_dir
        self.processor = processor

        self.img_id_to_anns = {}
        for ann in self.coco["annotations"]:
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.id2label = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.label2id = {c["name"]: c["id"] for c in self.coco["categories"]}

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, idx):
        img_info = self.coco["images"][idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        anns = self.img_id_to_anns.get(img_id, [])

        boxes = []
        class_labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]

            x = max(0, min(float(w), float(x)))
            y = max(0, min(float(h), float(y)))
            bw = max(0, min(float(w - x), float(bw)))
            bh = max(0, min(float(h - y), float(bh)))

            if bw <= 0 or bh <= 0:
                continue

            boxes.append([x, y, x + bw, y + bh])
            class_labels.append(int(ann["category_id"]))
            areas.append(float(bw * bh))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        annotations = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": boxes[i],
                    "category_id": class_labels[i],
                    "area": areas[i],
                    "iscrowd": iscrowd[i],
                }
                for i in range(len(boxes))
            ],
        }

        encoding = self.processor(
            images=image,
            annotations=annotations,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


class TATRCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_h = max(p.shape[-2] for p in pixel_values)
        max_w = max(p.shape[-1] for p in pixel_values)

        padded = []
        for p in pixel_values:
            c, h, w = p.shape
            out = torch.zeros((c, max_h, max_w), dtype=p.dtype)
            out[:, :h, :w] = p
            padded.append(out)

        return {
            "pixel_values": torch.stack(padded),
            "labels": labels,
        }


def main():
    print("=== TATR 100-ann Fine-tuning ===")
    print("Model:", MODEL_NAME)
    print("Train annotations:", TRAIN_ANN)
    print("Val annotations:", VAL_ANN)
    print("Train images:", TRAIN_IMG_DIR)
    print("Val images:", VAL_IMG_DIR)
    print("Output:", OUT_DIR)
    print(f"Hyperparams: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
    )

    print("Pretrained id2label:", model.config.id2label)

    train_ds = TATRFineTuneDataset(TRAIN_ANN, TRAIN_IMG_DIR, processor)
    val_ds = TATRFineTuneDataset(VAL_ANN, VAL_IMG_DIR, processor)

    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))
    print("Train id2label:", train_ds.id2label)

    collator = TATRCollator(processor)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=FP16,
        bf16=BF16,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        report_to="tensorboard",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = f"{OUT_DIR}/final"
    print("\nSaving final model to", final_dir)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print("Done.")


if __name__ == "__main__":
    main()
