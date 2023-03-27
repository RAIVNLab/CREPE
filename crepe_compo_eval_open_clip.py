import os
import subprocess
import argparse
    
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

def get_args():
    parser = argparse.ArgumentParser(description='Run image2text retrieval eval.')
    parser.add_argument('--compo-type', required=True, type=str, default='systematicity')
    parser.add_argument('--input-dir', required=True, type=str, default='/vision/group/CLIPComp/crepe/syst_hard_negatives')
    parser.add_argument('--train-dataset', type=str, default='cc12m')
    parser.add_argument('--model-dir', type=str, default='/vision/group/clip')
    parser.add_argument('--csv-caption-key', type=str, default='caption')
    parser.add_argument('--csv-img-key', type=str, default='image_id')
    parser.add_argument('--hard-neg-type', type=str, default='atom')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    return args

def gather_params(args, model, split):
    if args.compo_type == 'systematicity':
        if args.hard_neg_type == 'atom':
            hard_neg_key = 'valid_hard_negs_atom'
        elif args.hard_neg_type == 'comp':
            hard_neg_key = 'valid_hard_negs_comp'
        else:
            raise NotImplementedError
        
        retrieval_data_path = os.path.join(args.input_dir, f'syst_vg_hard_negs_{split}_in_{args.train_dataset}.csv')
        re_base = os.path.basename(retrieval_data_path)[:-4]
        re_base_with_type = re_base[:17] + '_' + args.hard_neg_type + re_base[17:]
        output_file = f'{model}_{re_base_with_type}_metrics.pkl'
        
    elif args.compo_type == 'productivity':
        hard_neg_key = 'hard_negs'
        if args.hard_neg_type in ['atom', 'negate', 'swap']:
            input_dir = os.path.join(args.input_dir, args.hard_neg_type)
            retrieval_data_path = os.path.join(input_dir, f'prod_vg_hard_negs_{args.hard_neg_type}_complexity_{split}.csv')
            re_base = os.path.basename(retrieval_data_path)[:-4]
            output_file = f'{args.train_dataset}_{model}_{re_base}_metrics.pkl'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.train_dataset)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    params = {
        'val-data': retrieval_data_path,
        'model': model,
        'one2many': True, # each image query has a different retrieval set with multiple texts
        'crop': True,
        'pretrained': os.path.join(args.model_dir, DATA2MODEL[args.train_dataset][model]),
        'csv-img-key': args.csv_img_key,
        'csv-caption-key': args.csv_caption_key,
        'hard-neg-key': hard_neg_key,
        'output-pkl': os.path.join(output_dir, output_file) if args.output_dir else None,
        'batch-size': 1 # this is set to 1 as each image query has a different retrieval set
        }
    return params

def run_eval_with_params(params):
    arguments = ' '.join(['--' + k + ' ' + str(params[k]) for k in params])
    train = 'python -m evaluate.main'
    train += ' ' + arguments
    print(train)
    subprocess.call(train, shell=True)

def main():
    args = get_args()
    models = DATA2MODEL[args.train_dataset].keys()
    if args.compo_type == 'systematicity':
        splits = COMPO_SPLITS
    elif args.compo_type == 'productivity':
        splits = COMPLEXITIES
    for model in models:
        for split in splits:
            params = gather_params(args, model, split)
            print('\n' + '*' * 45  + f' Evaluating {model} {args.compo_type} on split {split} ' + '*' * 45  + '\n')
            run_eval_with_params(params)

if __name__ == "__main__":
    main()
