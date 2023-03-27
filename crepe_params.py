import argparse

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/vision/group/CLIPComp/crepe/prod_hard_negatives/')
    parser.add_argument('--output-dir', type=str, default='prod_hard_negative_res')
    parser.add_argument('--hard-neg-type', type=str)
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="image_id",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="caption",
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
    # For CLIP & CyCLIP
    parser.add_argument("--model-name", type = str, default = "RN50")
    # For CyCLIP
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    args = parser.parse_args()
    return args