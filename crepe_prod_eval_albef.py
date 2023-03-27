# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from PIL import Image
from time import time
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

# ALBEF:
# from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
import ruamel.yaml as yaml
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
# from transformers import BertTokenizer
from models.tokenization_bert import BertTokenizer

from crepe_eval_utils import BaseCsvDataset, get_one2many_rank, get_one2many_metrics, DataInfo
from crepe_params import setup_args

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

max_text_length = 512
TEXT_DEFAULT_TOKENIZER = "bert-base-uncased"
text_tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)

def collator(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    texts = torch.cat([x[1] for x in batch], dim=0)
    masks = torch.cat([x[2] for x in batch], dim=0)

    return images, texts, masks

### DATASET CONSTRUCTION

def default_text_transform(texts):
    # Expect a list of texts
    tokenized_texts = []
    attention_masks = []
    start_time = time()
    for text in texts:
        tokenized = text_tokenizer(text, padding="max_length",
                                   max_length=max_text_length, truncation=True, return_tensors='pt')

        tokenized_texts.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])

    tokenized_texts = torch.cat(tokenized_texts, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return tokenized_texts, attention_masks

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args, config):
        super().__init__(input_filename, args)

        # albef transform:
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])   
        self.image_transform = test_transform
        self.text_transform = default_text_transform

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])
        image = self.transforms(raw_image)
        texts, attn_mask = self.text_transform([str(self.captions[idx])] + list(self.hard_negs[idx]))

        return image, texts, attn_mask

def get_data(args, retrieval_data_path, config):
    # Get CSVDataset
    input_filename = retrieval_data_path
    dataset = CsvDataset(
        input_filename,
        args,
        config=config)
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
            images, texts, masks = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)
                
            if one2many:
                image_feat = model.visual_encoder(images)        
                image_embed = model.vision_proj(image_feat[:,0,:])            
                image_embed = F.normalize(image_embed,dim=-1)

                text_out = model.text_encoder(texts, attention_mask = masks, mode='text')  
                text_feat = text_out.last_hidden_state
                text_emb = F.normalize(model.text_proj(text_feat[:,0,:]))

                for j in range(image_embed.shape[0]):
                    curr_image_emb = image_embed[j:j+1, :]
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
    output_dir = os.path.join(args.output_dir, 'albef')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # LOAD ALBEF
    config_str = './configs/Retrieval_coco.yaml'
    config = yaml.load(open(config_str, 'r'), Loader=yaml.Loader)
    tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
    albef = ALBEF(config=config, text_encoder=TEXT_DEFAULT_TOKENIZER, tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # MODEL CHECKPOINT
    checkpoint = torch.load('./ALBEF.pth', map_location='cpu') 
    state_dict = checkpoint['model']
    
    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],albef.visual_encoder)         
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],albef.visual_encoder_m)   
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
    
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.','')         
            state_dict[encoder_key] = state_dict[key] 
            del state_dict[key]                
    msg = albef.load_state_dict(state_dict,strict=False)  
    albef = albef.to(device)
    albef.eval()

    # Iterate over each complexity
    for i in range(4, 13):
        print('\n' + '*' * 45  + f' Evaluating on complexity {i} ' + '*' * 45  + '\n')
        start_time = time()
        retrieval_data_path = os.path.join(args.input_dir, f'{args.hard_neg_type}/prod_vg_hard_negs_{args.hard_neg_type}_complexity_{i}.csv')
        output_file = f'albef_complexity_{i}_metrics_{args.hard_neg_type}_neg.pkl'
        output_path = os.path.join(output_dir, output_file)
        
        data = get_data(args, retrieval_data_path, config)
        evaluate(albef, data, i, args.negative_type, output_path)

        print(f'Complexity {i} took {time() - start_time} seconds')

if __name__ == "__main__":
    main()
