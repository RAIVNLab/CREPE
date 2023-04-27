import ast
import logging
import os
from PIL import Image
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

### DATASET CONSTRUCTION

class BaseCsvDataset(Dataset):
    def __init__(self, input_filename, args, transforms=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename)
        # print(f"Total number of examples: {len(df)}.")
        self.crop = args.crop
        if self.crop:
            assert 'x' in df.columns and 'y' in df.columns and 'width' in df.columns and 'height' in df.columns, "missing x, y, width, or height."
            self.xs = df['x'].tolist()
            self.ys = df['y'].tolist()
            self.heights = df['height'].tolist()
            self.widths = df['width'].tolist()
        # print("cropping:", self.crop)
        self.one2many = args.one2many
        # print("one2many:", self.one2many)
        if self.one2many:
            self.hard_negs = [ast.literal_eval(ls_str) for ls_str in df[args.hard_neg_key]]
        self.images = df[args.csv_img_key].tolist()
        self.captions = df[args.csv_caption_key].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.captions)

    def get_image_by_id(self, image_id): 
        vg_image_paths = ['/nlp/scr/irena/data/visual_genome/img/VG_100K', '/nlp/scr/irena/data/visual_genome/img/VG_100K_2']
        for p in vg_image_paths:
            path = os.path.join(p, f"{image_id}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f'The image with id {image_id} is not found.')

    def __getitem__(self, idx):
        print("Not yet implemented.")
        assert(False)

@dataclass
class DataInfo:
    dataloader: DataLoader

# EVALUATION UTILITIES

def get_one2many_rank(image_features, text_features):
    logits_per_image = (image_features @ text_features.t()).detach().cpu()
    ground_truth = 0 # because the grountruth caption is placed first, see CsvDataset.__getitem__() in data.py
    ranking = torch.argsort(logits_per_image, descending=True)
    pred = torch.where(ranking == ground_truth)[1].detach().cpu().numpy()
    return pred

def get_one2many_metrics(preds, name='image_to_text'):
    metrics = {}
    metrics[f"{name}_mean_rank"] = preds.mean() + 1
    metrics[f"{name}_rank_std"] = preds.std()
    metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1

    for k in [1, 3, 5, 10]:
        metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        metrics[f"{name}_R@{k}_std"] = np.std(preds < k)
    return metrics

def get_metrics(image_features, text_features):
    metrics = {}
    logits_per_image = (image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1

        for k in [1, 3, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
