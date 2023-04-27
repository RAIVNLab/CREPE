# CREPE 

In this repository, you can find the code we used to evaluate these models: [open_clip](https://github.com/mlfoundations/open_clip) CLIP models, 
the official OpenAI CLIP models, CyCLIP, FLAVA and ALBEF on compositional reasoning in our paper [CREPE: Can Vision-Language Foundation Models Reason Compositionally?](https://arxiv.org/abs/2212.07796).

<img src="https://user-images.githubusercontent.com/36333627/228138304-86721dc2-3b8b-4c6e-acde-6bcfcd416cfe.png" width="800">

## Systematicity procedure
<img src="https://user-images.githubusercontent.com/36333627/228139225-71dc2cda-a5e7-4710-a9e8-7fbb8281eb01.png"  width="800">

## Produtivity procedure
<img src="https://user-images.githubusercontent.com/36333627/228139880-748609e2-9476-4c33-bca3-200deea369fe.png"  width="800">

## Evaluation instructions
In `crepe_eval_utils.py`, you can find common evaluation util functions, and you will need to replace `vg_image_paths` 
with the path to Visual Genome images on your machine. The VG images can be downloaded [here](https://drive.google.com/drive/folders/11dMtJByk7zmbQjV47PXVwfmakN3Gr5Ic?usp=share_link).

We evaluated all models on an NVIDIA TITAN X GPU with a CUDA version of 11.4.

### Evaluate open_clip CLIP models on systematicity and productivity
You will need to install the packages required to use open_clip [here](https://github.com/mlfoundations/open_clip/blob/main/requirements.txt). 
You can download the [pretrained CLIP models](https://github.com/mlfoundations/open_clip#pretrained-model-details) and replace `--model-dir` 
with your own model checkpoint directory path in `crepe_compo_eval_open_clip.py`. (You can also modify the code to use open_clip's 
[pretrained model interface](https://github.com/mlfoundations/open_clip#pretrained-model-interface).)

To evaluate all models reported in our paper, simply run:

```
python -m crepe_compo_eval_open_clip --compo-type <compositionality_type> --hard-neg-types <negative_type_1> <negative_type_2> --input-dir <path_to_crepe/crepe/syst_hard_negatives>  --output-dir <log_directory>
```

where the valid compositionality types are `systematicity` and `productivity`. The valid negative types are `atom`, `comp` and `combined` (`atom`+`comp`) for systematicity, and `atom`, `swap` and `negate` for productivity.

To evaluate other pretrained models, simply modify the `--train-dataset` argument and/or the `DATA2MODEL` variable in  `crepe_compo_eval_open_clip.py`. 
**Note that the systematicity eval set should only be used to evaluate models pretrained on CC12M, YFCC15M or LAION400M.**

### Evaluate all other vision-language models on productivity
For each model, you will need to clone the model's official repository, set up
an environment according to its instructions and place the files `crepe_prod_eval_<model>.py`
and `crepe_eval_utils.py` to their relevant locations. In `crepe_params.py`, you will need to replace `--input-dir`
with your own directory path to CREPE's productivity hard negatives test set.

#### CLIP-specific instructions
Clone the CLIP repository [here](https://github.com/openai/CLIP) and place `crepe_prod_eval_clip.py`
and `crepe_eval_utils.py` on the top level of the repository. To evaluate models, simply run:

```
python -m crepe_prod_eval_clip --model-name <model_name> --hard-neg-types <negative_type> --output-dir <log_directory> 
```

where the valid negative types are `atom`, `swap` and `negate`, and model names are `RN50`, `RN101`, `ViT-B/32`, `ViT-B/16` and `ViT-L/14`.

#### CyCLIP-specific instructions
Clone the CyCLIP repository [here](https://github.com/goel-shashank/CyCLIP), place `crepe_prod_eval_cyclip.py`
and `crepe_eval_utils.py` on the top level of the repository and download the
model checkpoint under the folder `cyclip.pt` (accessible from the bottom of the
repository's README). To evaluate models, simply run:

```
python -m crepe_prod_eval_cyclip --hard-neg-types <negative_type> --output-dir <log_directory>
```

#### FLAVA-specific instructions
Clone the FLAVA repository [here](https://github.com/facebookresearch/multimodal) and copy `crepe_prod_eval_flava.py`
and `crepe_eval_utils.py` into the folder `examples/flava/`. To evaluate models, simply run:

```
python -m crepe_prod_eval_flava --hard-neg-types <negative_type> --output-dir <log_directory>
```

#### ALBEF-specific instructions
Clone the ALBEF repository [here](https://github.com/salesforce/ALBEF/tree/b9727e43c3040491774d1b22cc27718aa7772fac),
copy `crepe_prod_eval_albef.py` and `crepe_eval_utils.py` to the top level of the repository
and download the pretrained checkpoint marked '14M' from the repository. To evaluate models, simply run:

```
python -m crepe_prod_eval_albef --hard-neg-types <negative_type> --output-dir <log_directory>
```

## Citation
If you find our work helpful, please cite us:

```bibtex
 @article{ma2023crepe,
  title={CREPE: Can Vision-Language Foundation Models Reason Compositionally?}, 
  author={Zixian Ma and Jerry Hong and Mustafa Omer Gul and Mona Gandhi and Irena Gao and Ranjay Krishna},
  year={2023},
  journal={arXiv preprint arXiv:2212.07796},
}
```
