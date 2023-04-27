import os
import json
import logging
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from open_clip import tokenize, create_model_and_transforms
from crepe_eval_utils import BaseCsvDataset, get_one2many_metrics, get_one2many_rank, get_metrics
from crepe_params import setup_args
    
DATA2MODEL = {
    'cc12m': {
        'RN50-quickgelu': 'rn50-quickgelu-cc12m-f000538c.pt'
    },
    'yfcc': {
        'RN50-quickgelu': 'rn50-quickgelu-yfcc15m-455df137.pt', 
        'RN101-quickgelu': 'rn101-quickgelu-yfcc15m-3e04b30e.pt'
    },
    'laion': {
        'ViT-B-16':'vit_b_16-laion400m_e32-55e67d44.pt',
        'ViT-B-16-plus-240': 'vit_b_16_plus_240-laion400m_e32-699c4b84.pt',
        'ViT-B-32-quickgelu': 'vit_b_32-quickgelu-laion400m_e32-46683a32.pt',
        'ViT-L-14': 'vit_l_14-laion400m_e32-3d133497.pt',
    }
}

COMPO_SPLITS = ['seen_compounds', 'unseen_compounds']
COMPLEXITIES = list(range(4, 13))

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args, transforms):
        super().__init__(input_filename, args, transforms=transforms)

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])
        image = self.transforms(raw_image)
        if self.one2many:
            texts = tokenize([str(self.captions[idx])] + list(self.hard_negs[idx]))
        else:
            texts = tokenize([str(self.captions[idx])])[0]
        return image, texts

def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        args, 
        preprocess_fn) 
    num_samples = len(dataset)

    sampler = None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
    
def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["val"] = get_csv_dataset(
        args, preprocess_val, is_train=False)
    return data

def evaluate(model, data, args):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    autocast = torch.cuda.amp.autocast
    dataloader = data['val'].dataloader

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    all_image_features, all_text_features = [], []
    one2many = dataloader.dataset.one2many
    if one2many:
        all_ranks = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            
            if one2many:
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                texts = torch.squeeze(texts, dim=0)
                text_features = model.encode_text(texts)
                text_features = F.normalize(text_features, dim=-1)

                rank = get_one2many_rank(image_features, text_features)
                all_ranks.append(rank)
            else:
                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())

        if one2many:
            val_metrics = get_one2many_metrics(np.array(all_ranks), arg=args)
            metrics.update(
                {**val_metrics}
            )
        else:
            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features)
            )
            metrics.update(
                {**val_metrics}
            )

    logging.info("\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))
    
    return metrics

def gather_params(args, hard_neg_type, split):
    if args.compo_type == 'systematicity':
        if hard_neg_type in ['atom', 'comp', 'combined']:
            hard_neg_key = f'valid_hard_negs_{hard_neg_type}'
        else:
            raise NotImplementedError
        
        retrieval_data_path = os.path.join(args.input_dir, f'syst_vg_hard_negs_{split}_in_{args.train_dataset}.csv')
        
    elif args.compo_type == 'productivity':
        hard_neg_key = 'hard_negs'
        if hard_neg_type in ['atom', 'negate', 'swap']:
            input_dir = os.path.join(args.input_dir, hard_neg_type)
            retrieval_data_path = os.path.join(input_dir, f'prod_vg_hard_negs_{hard_neg_type}_complexity_{split}.csv')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    args.val_data = retrieval_data_path
    args.one2many = True
    args.crop = True
    args.hard_neg_key = hard_neg_key
    args.batch_size = 1
    return args

def main():
    args = setup_args()
    models = DATA2MODEL[args.train_dataset].keys()
    if args.compo_type == 'systematicity':
        splits = COMPO_SPLITS
    elif args.compo_type == 'productivity':
        splits = COMPLEXITIES
    
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)

    for model_name in models:
        pretrained = os.path.join(args.model_dir, DATA2MODEL[args.train_dataset][model_name])
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            pretrained,
            precision='amp',
            device=device
        )
        for hard_neg_type in args.hard_neg_types:
            all_metrics = {}
            for split in splits:
                # params = gather_params(args, model, split)
                print('\n' + '*' * 45  + f' Evaluating {model_name} {args.compo_type} on HN-{hard_neg_type.upper()} test set split {split} ' + '*' * 45  + '\n')
                args = gather_params(args, hard_neg_type, split)
                # initialize datasets
                data = get_data(args, (preprocess_train, preprocess_val))
                assert len(data), 'At least one dataset must be specified.'

                metrics = evaluate(model, data, args)

                all_metrics[split] = metrics

            if args.output_dir:
                output = os.path.join(args.output_dir, f'{args.compo_type}_{args.train_dataset}_{model_name}_{hard_neg_type}_metrics.json')
                print("saving results to:", output)
                with open(output, 'w') as f:
                    json.dump(all_metrics, f)

if __name__ == "__main__":
    main()
