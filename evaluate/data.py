from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from open_clip import tokenize
from crepe_eval_utils import BaseCsvDataset

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

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
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
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
        args, preprocess_val, is_train=False)
    return data
