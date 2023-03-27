import logging
import os
from time import time
import pickle

import torch
import torchvision.transforms.functional as TF
import clip
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd

from crepe_eval_utils import BaseCsvDataset, get_one2many_rank, get_one2many_metrics, DataInfo
from crepe_params import setup_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def collator(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    texts = torch.cat([x[1] for x in batch], dim=0)

    return images, texts

### DATASET CONSTRUCTION

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args, processor):
        super().__init__(input_filename, args)

        self.processor = processor
        self.device = device

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])

        image = self.processor(raw_image)
        texts = self.process_text([str(self.captions[idx])] + list(self.hard_negs[idx]))
        return image, texts

    def process_text(self, texts):
        proc_text = [clip.tokenize(text, truncate=True) for text in texts]
        return torch.cat(proc_text)

def get_data(args, retrieval_data_path, processor, device):
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

def evaluate(model, data, complexity, negative_type, output_path, device):
    metrics = {}

    dataloader = data.dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    cumulative_loss = 0.0
    all_image_features, all_text_features = [], []
    one2many = dataloader.dataset.one2many

    if one2many:
        all_ranks = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            if one2many:
                image_emb = model.encode_image(images)
                image_emb /= image_emb.norm(dim = -1, keepdim = True)
                
                text_emb = model.encode_text(texts)
                text_emb /= text_emb.norm(dim = -1, keepdim = True)

                for j in range(image_emb.shape[0]):
                    curr_image_emb = image_emb[j:j+1, :]
                    curr_text_emb = text_emb[j*6:(j+1)*6, :]
                    rank = get_one2many_rank(curr_image_emb, curr_text_emb)
                    all_ranks.append(rank)

            print(f'Processed example {i*16}')

    metrics = get_one2many_metrics(np.array(all_ranks))

    # Alter output here
    logging.info(
        "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    # Dump metrics as pickle file
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(metrics, f)
    except Exception as e:
        raise(e)

    return metrics

def main():
    args = setup_args()
    output_dir = os.path.join(args.output_dir, 'open_ai_clip')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(name = args.model_name, device=device)
    model = model.to(device)
    model.eval()

    # Iterate over each complexity
    for i in range(4, 13):
        print('\n' + '*' * 45  + f' Evaluating on complexity {i} ' + '*' * 45  + '\n')
        start_time = time()
        retrieval_data_path = os.path.join(args.input_dir, f'{args.hard_neg_type}/prod_vg_hard_negs_{args.hard_neg_type}_complexity_{i}.csv')

        if args.model_name == "RN50" or args.model_name == "RN101":
            model_save_name = args.model_name
        elif args.model_name == "ViT-B/32":
            model_save_name = 'vit_b32'
        elif args.model_name == "ViT-B/16":
            model_save_name = 'vit_b16'
        elif args.model_name == "ViT-L/14":
            model_save_name = 'vit_l14'

        output_file = f'open_ai_clip_{model_save_name}_complexity_{i}_metrics_{args.hard_neg_type}_neg.pkl'
        output_path = os.path.join(output_dir, output_file)

        data = get_data(args, retrieval_data_path, preprocess, device)
        evaluate(model, data, i, args.negative_type, output_path, device)

        print(f'Complexity {i} took {time() - start_time} seconds')

if __name__ == '__main__':
    main()
