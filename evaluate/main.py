import random
import numpy as np
import logging
import pickle
import torch
import torch.nn.functional as F

from contextlib import suppress
from open_clip import create_model_and_transforms

from evaluate.data import get_data
from evaluate.params import parse_args
from crepe_eval_utils import get_one2many_metrics, get_one2many_rank, get_metrics

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def evaluate(model, data, epoch, args):
    metrics = {}
    device = torch.device(args.device)
    model.eval()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    dataloader = data['val'].dataloader
    num_samples = 0
    samples_per_val = dataloader.num_samples

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    cumulative_loss = 0.0
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
                image_features = model(images, None)
                image_features /= image_features.norm(dim = -1, keepdim = True)
                for text in texts:
                    text_features = model(None, text)
                    text_features /= text_features.norm(dim = -1, keepdim = True)
                rank = get_one2many_rank(image_features, text_features, model.logit_scale.exp().mean().cpu())
                all_ranks.append(rank)
            else:
                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

        if one2many:
            val_metrics = get_one2many_metrics(np.array(all_ranks), arg=args)
            metrics.update(
                {**val_metrics}
            )
        else:
            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
                arg=args
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
        if args.output_pkl:
            with open(args.output_pkl, 'wb') as file:
                pickle.dump(val_metrics, file)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    print(metrics)
    return metrics

def main():
    args = parse_args()

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    )

    start_epoch = 0
    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    evaluate(model, data, start_epoch, args)

if __name__ == "__main__":
    main()
