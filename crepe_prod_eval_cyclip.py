# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import argparse
import logging
import os
from PIL import Image, ImageFile
from dataclasses import dataclass
from time import time
import json

import torch
import torchvision.transforms.functional as TF
from pkgs.openai.clip import load
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pandas as pd

from crepe_eval_utils import BaseCsvDataset, get_one2many_rank, get_one2many_metrics, DataInfo
from crepe_params import setup_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def collator(batch):
    texts = []

    images = torch.stack([x[0] for x in batch], dim=0)
    texts = torch.cat([x[1] for x in batch], dim=0)
    attention_masks = torch.cat([x[2] for x in batch], dim=0)

    return images, texts, attention_masks

### DATASET CONSTRUCTION

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args, processor):
        super().__init__(input_filename, args)

        self.processor = processor

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])
        

        image = torch.tensor(self.processor.process_image(raw_image))
        return_dict = self.processor.process_text([str(self.captions[idx])] + list(self.hard_negs[idx]))
        input_ids = return_dict['input_ids']
        attention_mask = return_dict['attention_mask']

        return image, input_ids, attention_mask

def get_data(args, retrieval_data_path, processor):
    # Get CSVDataset
    input_filename = retrieval_data_path
    dataset = CsvDataset(
        input_filename,
        args,
        processor)
    num_samples = len(dataset)
    sampler = None
    shuffle=False

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        collate_fn=collator
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader)

### EVALUATION

def evaluate(model, data, complexity, negative_type):
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    dataloader = data.dataloader
    # num_samples = 0
    # samples_per_val = dataloader.num_samples

    # cumulative_loss = 0.0
    # all_image_features, all_text_features = [], []
    one2many = dataloader.dataset.one2many

    if one2many:
        all_ranks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts, attention_mask = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            attention_mask = attention_mask.to(device=device, non_blocking=True)
                
            if one2many:
                image_emb = model.get_image_features(images)
                image_emb /= image_emb.norm(dim = -1, keepdim = True)

                text_emb = model.get_text_features(input_ids = texts, attention_mask = attention_mask)
                text_emb /= text_emb.norm(dim = -1, keepdim = True)

                set_size = text_emb.shape[0] // image_emb.shape[0]
                for j in range(image_emb.shape[0]):
                    curr_image_emb = image_emb[j:j+1, :]
                    curr_text_emb = text_emb[j*set_size:(j+1)*set_size, :]
                    rank = get_one2many_rank(curr_image_emb, curr_text_emb)
                    all_ranks.append(rank)

            print(f'Processed example {i*16}')

    metrics = get_one2many_metrics(np.array(all_ranks))

    # Alter output here
    logging.info(
        "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )     

    return metrics
    
def main():
    args = setup_args()
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, 'cyclip')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load(name = args.model_name, pretrained = args.pretrained)
    checkpoint = torch.load('best.pt', map_location=device)
    state_dict = checkpoint['state_dict']
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    for hard_neg_type in args.hard_neg_types:
        all_metrics = {}
        # Iterate over each complexity
        for i in range(4, 13):
            print('\n' + '*' * 45  + f' Evaluating on complexity {i} ' + '*' * 45  + '\n')
            start_time = time()
            retrieval_data_path = os.path.join(args.input_dir, f'{hard_neg_type}/prod_vg_hard_negs_{hard_neg_type}_complexity_{i}.csv')
            
            data = get_data(args, retrieval_data_path, processor)
            metrics = evaluate(model, data, i, hard_neg_type)

            print(f'Complexity {i} took {time() - start_time} seconds')

            all_metrics[i] = metrics

        if args.output_dir:
            output = os.path.join(output_dir, f'productivity_cyclip_{args.model_name}_{hard_neg_type}_metrics.json')
            print("saving results to:", output)
            with open(output, 'w') as f:
                json.dump(all_metrics, f)

if __name__ == "__main__":
    main()
