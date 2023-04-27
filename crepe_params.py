import argparse

def setup_args():
    parser = argparse.ArgumentParser(description="Run image2text retrieval eval.")
    parser.add_argument("--compo-type", required=True, type=str, default="systematicity", help="Either systematicity or productivity")
    parser.add_argument("--input-dir", required=True, type=str, default="/vision/group/CLIPComp/crepe/prod_hard_negatives")
    parser.add_argument('--hard-neg-types', required=True, type=str, nargs='+', help="The type(s) of hard negatives to include in the retrieval set.")
    parser.add_argument("--model-dir", type=str, default="/vision/group/clip")
    parser.add_argument("--output-dir", type=str, default="log/")
    parser.add_argument("--csv-img-key", type=str, default="image_id")
    parser.add_argument("--csv-caption-key", type=str, default="caption")
    parser.add_argument("--hard-neg-key", type=str, default="hard_negs", help="The column name of the hard negative captions.")
    parser.add_argument("--crop", type=bool, default=True, help="Whether to crop the image input.")
    parser.add_argument("--one2many", type=bool, default=True, help="Whether each image query has a different retrieval text set.")
    # For systematicity eval on open_clip's pretrained models with known training dataset
    parser.add_argument("--train-dataset", type=str, default="cc12m")
    # For CLIP & CyCLIP
    parser.add_argument("--model-name", type=str, default="RN50")
    # For CyCLIP
    parser.add_argument("--pretrained", default=False, action="store_true", help="Use the OpenAI pretrained models")
    args = parser.parse_args()
    return args