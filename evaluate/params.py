import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--hard-neg-key",
        type=str,
        default="hard_negs",
        help="For csv-like datasets, the name of the key for the hard negative captions."
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="For evaluation that starts with a certain id in the input data."
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=0,
        help="For evaluation that ends with a certain id in the input data."
    )
    parser.add_argument(
        "--crop",
        type=bool,
        default=True,
        help="Whether to crop the image input.",
    )
    parser.add_argument(
        "--one2many",
        type=bool,
        default=True,
        help="Whether each image query has a different retrieval text set.",
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=None,
        help="Path to saved metrics.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )

    args = parser.parse_args()

    return args
