# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast

import logging
import os
from PIL import Image
from dataclasses import dataclass
from time import time
import pickle

import torch
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmultimodal.models.flava.model import flava_model
from transformers import BertTokenizer
import torchvision.transforms.functional as TF
import numpy as np

import pandas as pd

from crepe_eval_utils import BaseCsvDataset, get_one2many_rank, get_one2many_metrics, DataInfo
from crepe_params import setup_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

max_text_length = 512
TEXT_DEFAULT_TOKENIZER = "bert-base-uncased"
text_tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)

def collator(batch):
    texts = []
    images = torch.stack([x[0]["image"] for x in batch], dim=0)
    texts = torch.cat([x[1] for x in batch], dim=0)

    return images, texts

### DATASET CONSTRUCTION

def default_text_transform(texts):
    # Expect a list of texts
    tokenized_texts = []
    start_time = time()
    for text in texts:
        tokenized = text_tokenizer(text, padding="max_length",
                                   max_length=max_text_length, truncation=True, return_tensors='pt')
        tokenized_texts.append(torch.LongTensor(tokenized['input_ids']))
    tokenized_texts = torch.cat(tokenized_texts, dim=0)

    return tokenized_texts    

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args):
        super().__init__(input_filename, args)

        self.image_transform = FLAVAImageTransform(is_train=False)
        self.text_transform = default_text_transform

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])
        image = self.image_transform(raw_image)
        if self.one2many:
            texts = self.text_transform([str(self.captions[idx])] + list(self.hard_negs[idx]))
        else:
            texts = self.text_transform([str(self.captions[idx])])[0]
        return image, texts

def get_data(args, retrieval_data_path):
    # Get CSVDataset
    input_filename = retrieval_data_path
    dataset = CsvDataset(
        input_filename,
        args)
    num_samples = len(dataset)
    sampler = None
    shuffle=False

    dataloader = DataLoader(
        dataset,
        batch_size=8,
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

def evaluate(model, data, complexity, negative_type, output_path):
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    dataloader = data.dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    cumulative_loss = 0.0
    all_image_features, all_text_features = [], []
    one2many = dataloader.dataset.one2many
    assert(one2many, "Not one2many?")

    if one2many:
        all_ranks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
                
            if one2many:
                _, image_emb = model.encode_image(images, projection=True)
                image_emb = nn.functional.normalize(image_emb, dim=-1)
                _, text_emb = model.encode_text(texts, projection=True)
                text_emb = nn.functional.normalize(text_emb)

                for j in range(image_emb.shape[0]):
                    curr_image_emb = image_emb[j:j+1, :]
                    curr_text_emb = text_emb[j*6:(j+1)*6, :]
                    rank = get_one2many_rank(curr_image_emb, curr_text_emb)
                    all_ranks.append(rank)

            # print(f'Processed example {i*8}')

    metrics = get_one2many_metrics(np.array(all_ranks))

    # Alter output here
    logging.info(
        "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    print(metrics)
    # Dump metrics as pickle file
    try:
        with open(output_path, 'wb') as f:                                                                                                                   
            pickle.dump(metrics, f)                                                                                                                          
    except Exception as e:                                                                                                                                
        raise(e)        
    
    return metrics
    
def main():
    args = setup_args()
    output_dir = os.path.join(args.output_dir, 'flava')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the model
    flava = flava_model(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    flava = flava.to(device)
    flava.eval()

    # Iterate over each complexity
    for i in range(4, 13):
        print('\n' + '*' * 45  + f' Evaluating on complexity {i} ' + '*' * 45  + '\n')
        start_time = time()
        retrieval_data_path = os.path.join(args.input_dir, f'{args.hard_neg_type}/prod_vg_hard_negs_{args.hard_neg_type}_complexity_{i}.csv')
        output_file = f'flava_complexity_{i}_metrics_{args.hard_neg_type}_neg.pkl'
        output_path = os.path.join(output_dir, output_file)
        
        data = get_data(args, retrieval_data_path)
        evaluate(flava, data, i, args.hard_neg_type, output_path)

        print(f'Complexity {i} took {time() - start_time} seconds')

if __name__ == "__main__":
    main()
