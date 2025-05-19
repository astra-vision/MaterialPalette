<div align='center'>

## Material Palette: Extraction of Materials from a Single Image (CVPR 2024)

<div>
  <a href="https://wonjunior.github.io/">Ivan Lopes</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://fabvio.github.io/">Fabio Pizzati</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://team.inria.fr/rits/membres/raoul-de-charette/">Raoul de Charette</a><sup>1</sup>
  <br>
  <sup>1</sup> Inria,
  <sup>2</sup> Oxford Uni.
</div>
<br>

[![Project page](https://img.shields.io/badge/🚀_Project_Page-_-darkgreen?style=flat-square)](https://astra-vision.github.io/MaterialPalette/)
[![paper](https://img.shields.io/badge/paper-_-darkgreen?style=flat-square)](https://github.com/astra-vision/MaterialPalette/releases/download/preprint/material_palette.pdf)
[![cvf](https://img.shields.io/badge/CVF-_-darkgreen?style=flat-square)](https://openaccess.thecvf.com/content/CVPR2024/html/Lopes_Material_Palette_Extraction_of_Materials_from_a_Single_Image_CVPR_2024_paper.html)
[![dataset](https://img.shields.io/badge/🤗_dataset--darkgreen?style=flat-square)](https://huggingface.co/datasets/ilopes/texsd)
[![star](https://img.shields.io/badge/⭐_star--darkgreen?style=flat-square)](https://github.com/astra-vision/MaterialPalette/stargazers)
<!--[![arXiv](https://img.shields.io/badge/arXiv-_-darkgreen?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2311.17060)-->


<b>TL;DR,</b> Material Palette extracts a palette of PBR materials - <br>albedo, normals, and roughness - from a single real-world image.

</div>

https://github.com/astra-vision/MaterialPalette/assets/30524163/44e45e58-7c7d-49a3-8b6e-ec6b99cf9c62


<!--ts-->
* [Overview](#overview)
* [1. Installation](#1-installation)
* [2. Quick Start](#2-quick-start)
  * [Generation](#-generation)
  * [Complete Pipeline](#-complete-pipeline)
* [3. Project Structure](#3-project-structure)
* [4. (optional) Retraining](#4-optional-training)
* [Acknowledgments](#acknowledgments)
* [Licence](#license)
<!--te-->

<!--## 🚨 Todo

- 3D rendering script.-->

## Overview

This is the official repository of [**Material Palette**](https://astra-vision.github.io/MaterialPalette/). In a nutshell, the method works in three stages: first, concepts are extracted from an input image based on a user-provided mask; then, those concepts are used to generate texture images; finally, the generations are decomposed into SVBRDF maps (albedo, normals, and roughness). Visit our project page or consult our paper for more details!

![pipeline](https://github.com/astra-vision/MaterialPalette/assets/30524163/be03b0ca-bee2-4fc7-bebd-9519c3c4947d)

**Content**: This repository allows the extraction of texture concepts from image and region mask sets. It also allows generation at different resolutions. Finally, it proposes a decomposition step thanks to our decomposition model, for which we share the training weights.

> [!TIP]
> We propose a ["Quick Start"](#2-quick-start) section: before diving straight into the full pipeline, we share four pretrained concepts ⚡ so you can go ahead and experiment with the texture generation step of the method: see ["§ Generation"](#-generation). Then you can try out the full method with your own image and masks = concept learning + generation + decomposition, see ["§ Complete Pipeline"](#-complete-pipeline).


## 1. Installation

 1. Download the source code with git
    ```
    git clone https://github.com/astra-vision/MaterialPalette.git
    ```
    The repo can also be downloaded as a zip [here](https://github.com/astra-vision/MaterialPalette/archive/refs/heads/master.zip).

 2. Create a conda environment with the dependencies.
    ```
    conda env create --verbose -f deps.yml
    ```
    This repo was tested with [**Python**](https://www.python.org/doc/versions/) 3.10.8, [**PyTorch**](https://pytorch.org/get-started/previous-versions/) 1.13, [**diffusers**](https://huggingface.co/docs/diffusers/installation) 0.19.3, [**peft**](https://huggingface.co/docs/peft/en/install) 0.5, and [**PyTorch Lightning**](https://lightning.ai/docs/pytorch/stable/past_versions.html) 1.8.3.

 3. Load the conda environment:
    ```
    conda activate matpal
    ```

 4. If you are looking to perform decomposition, download our pre-trained model and untar the archive:
    ```
    wget https://github.com/astra-vision/MaterialPalette/releases/download/weights/model.tar.gz
    ```
    <sup>This is not required if you are only looking to perform texture extraction</sup>

<!--
In case you want to retrain the source model, you can download the AmbientCG samples using the following command (`outdir` is the directory where the dataset will be downloaded to):
```
python capture/data/download.py outdir
```-->

## 2. Quick start

Here are instructions to get you started using **Material Palette**. First, we provide some optimized concepts so you can experiment with the generation pipeline. We also show how to run the method on user-selected images and masks (concept learning + generation + decomposition)


### § Generation

| Input image | 1K | 2K | 4K | 8K | ⬇️ LoRA ~8Kb
| :-: | :-: | :-: | :-: | :-: | :-: |
| <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/ba3126d7-ce54-4895-8d59-93f1fd22e7d6" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/e1ec9c9e-d618-4314-82a3-2ac2432af668" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/d960a216-5558-4375-9bf2-5a648221aa55" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/45ad2ca9-8be7-48ba-b368-5528ae021627" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/c9140b16-a59f-4898-b49f-5c3635a3ea85" alt="J" width="100"/> | [![x](https://img.shields.io/badge/-⚡blue_tiles.zip-black)](https://github.com/astra-vision/MaterialPalette/files/14601640/blue_tiles.zip)
| <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/f5838959-aeeb-417a-8030-0fab5e39443b" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/4b756fae-3ea6-4d40-b4e6-0a8c50674e14" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/91aefd19-0985-4b84-81a2-152eb16b87e0" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/c9547e54-7bac-4f3d-8d94-acafd61847d9" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/069d639b-71bc-4f67-a735-a3b44d7bc683" alt="J" width="100"/> | [![x](https://img.shields.io/badge/-⚡cat_fur.zip-black)](https://github.com/astra-vision/MaterialPalette/files/14601641/cat_fur.zip)
| <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/b16bc25f-e5c5-45ad-bf3b-ef28cb57ed30" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/0ae31915-7bc5-4177-8b84-6988cccc2c24" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/e501c66d-a5b7-42e4-9ec2-0a12898280ed" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/290b685a-554c-4c62-ab0d-9d66a2945f09" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/378be48d-61e5-4a8a-b2cd-1002aec541bf" alt="J" width="100"/> | [![x](https://img.shields.io/badge/-⚡damaged.zip-black)](https://github.com/astra-vision/MaterialPalette/files/14601642/damaged.zip)
| <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/3c69d0c0-d91a-4d19-b0c0-b9dceb4477cf" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/ec6c62ea-00f7-4284-8cc3-6604159a3b5f" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/26c6ad3d-2306-4ad3-97a7-6713d5f4e5ee" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/94f7caa1-3ade-4b62-b0c6-b758a3a05d3f" alt="J" width="100"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/36630e65-9a2f-4a77-bb1b-0214d5f1b6f9" alt="J" width="100"/> | [![x](https://img.shields.io/badge/-⚡ivy_bricks.zip-black)](https://github.com/astra-vision/MaterialPalette/files/14601643/ivy_bricks.zip)

<sup>All generations were downscaled for memory constraints.</sup>


Go ahead and download one of the above LoRA concept checkpoints, example for "blue_tiles":

```
wget https://github.com/astra-vision/MaterialPalette/files/14601640/blue_tiles.zip;
unzip blue_tiles.zip
```
To generate from a checkpoint, use the [`concept`](./concept/) module either via the command line interface or the functional interface in python:
- ![](https://img.shields.io/badge/$-command_line-white?style=flat-square)
  ```
  python concept/infer.py path/to/LoRA/checkpoint
  ```
- ![](https://img.shields.io/badge/-python-white?style=flat-square&logo=python)
  ```
  import concept
  concept.infer(path_to_LoRA_checkpoint)
  ```

Results will be placed relative to the checkpoint directory in a `outputs` folder.

You have control over the following parameters:
- `stitch_mode`: concatenation, average, or weighted average (*default*);
- `resolution`: the output resolution of the generated texture;
- `prompt`: one of the four prompt templates:
  - `"p1"`: `"top view realistic texture of S*"`,
  - `"p2"`: `"top view realistic S* texture"`,
  - `"p3"`: `"high resolution realistic S* texture in top view"`,
  - `"p4"`: `"realistic S* texture in top view"`;
- `seed`: inference seed when sampling noise;
- `renorm`: whether or not to renormalize the generated samples generations based on input image (this option can only be used when called from inside the pipeline, *ie.* when the input image is available);
- `num_inference_steps`: number of denoising steps.

<sup>A complete list of parameters can be viewed with `python concept/infer.py --help`</sup>


### § Complete Pipeline

We provide an example (input image with user masks used for the pipeline figure). You can download it here: [**mansion.zip**](https://github.com/astra-vision/MaterialPalette/files/14619163/mansion.zip) (credits photograph:  [Max Rahubovskiy](https://www.pexels.com/@heyho/)).

To help you get started with your own images, you should follow this simple data structure: one folder per inverted image, inside should be the input image (`.jpg`, `.jpeg`, or `.png`) and a subdirectory named `masks` containing the different region masks as `.png` (these **must all have the same aspect ratio** as the RGB image). Here is an overview of our mansion example:
```
├── masks/
│ ├── wood.png
│ ├── grass.png
│ └── stone.png
└── mansion.jpg
```

|region|mask|overlay|generation|albedo|normals|roughness|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|![#6C8EBF](https://placehold.co/15x15/6C8EBF/6C8EBF.png) | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/23d422e4-6d69-4dd5-a823-44b284b1589d" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/84601e24-74d3-4da0-96e2-a3554f3481b4" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/918b2f5a-e975-444c-8a1b-523df9492eab" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/fa367b0c-5e22-4148-b785-23d147faead0" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/444a63b4-3eea-47de-9dac-1e9b122453a7" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/3a551001-249d-41ab-9e7a-a990950a8632" alt="J" height="85"/>|
|![#EDB01A](https://placehold.co/15x15/EDB01A/EDB01A.png) | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/712887ef-d235-433b-9e95-5bb58c5d96ee" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/56b8d10f-041f-414b-ba2c-6ea08cbdb2c2" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/fa0bad8d-1d9e-4019-9a99-14be732612b3" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/c1569bee-d387-4b60-b12b-9aadaf693dcc" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/d10bb074-a305-44c7-8536-a29b761ad14d" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/9ebed2ac-408d-4ad8-8ec4-99344e9ad85f" alt="J" height="85"/>|
|![#AA4A44](https://placehold.co/15x15/AA4A44/AA4A44.png) | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/ee22ad46-3f63-460a-ab8a-8b071cfd2b75" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/90a9904c-db25-4fec-a60c-46dcadf8de59" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/f8f9e7b3-e2f9-4603-823a-9f79cfe8d2a9" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/ede74224-80ae-4fe3-8eee-aae53935cc0e" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/3378dd1f-8801-47e8-a570-57e908f21e4d" alt="J" height="85"/>|<img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/77b31a23-5430-4221-a28a-c03e9099d45c" alt="J" height="85"/>|

<!-- | Input image | mask 1 | mask 2 | mask 3 | mask 4 |
| :-: | :-: | :-: | :-: | :-: |
| `bricks.jpg` | `runningbond.png` | `herringbone.png` | `basketweave.png` | `stonewall.png` |
| <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/55e89f01-81b5-4916-a817-c430eb70b12c" alt="J" width="150"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/0afa522a-207e-4762-b9fb-823736776458" alt="J" width="150"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/a90f5e0f-966f-475e-8e0d-dbd331960a5e" alt="J" width="150"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/182f4f15-b6aa-4888-87a7-59dc941d4688" alt="J" width="150"/> | <img src="https://github.com/astra-vision/MaterialPalette/assets/30524163/f68419e6-62ae-433d-9178-5e3e41417893" alt="J" width="150"/> -->


To invert and generate textures from a folder, use [`pipeline.py`](./pipeline.py):

- ![](https://img.shields.io/badge/$-command_line-white?style=flat-square)
  ```
  python pipeline.py path/to/folder
  ```

Under the hood, it uses two modules:
1. [`concept`](./concept), to extract and generate the texture ([`concept.crop`](./concept/crop.py), [`concept.invert`](./concept/invert.py), and [`concept.infer`](./concept/infer.py));
2. [`capture`](./capture/), to perform the BRDF decomposition.

A minimal example is provided here:

- ![](https://img.shields.io/badge/-python-white?style=flat-square&logo=python)
  ```
  ## Extract square crops from image for each of the binary masks located in <path>/masks
  regions = concept.crop(args.path)

  ## Iterate through regions to invert the concept and generate texture views
  for region in regions.iterdir():
      lora = concept.invert(region)
      concept.infer(lora, renorm=True)

  ## Construct a dataset with all generations and load pretrained decomposition model
  data = capture.get_data(predict_dir=args.path, predict_ds='sd')
  module = capture.get_inference_module(pt='model.ckpt')

  ## Proceed with inference on decomposition model
  decomp = Trainer(default_root_dir=args.path, accelerator='gpu', devices=1, precision=16)
  decomp.predict(module, data)
  ```
<sup>To view options available for the concept learning, use ``PYTHONPATH=. python concept/invert.py --help``</sup>

> [!IMPORTANT]
> By default, both `train_text_encoder` and `gradient_checkpointing` are set to `True`. Also, this implementation does not include the `LPIPS` filter/ranking of the generations. The code will only output a single sample per region. You may experiment with different prompts and parameters (see ["Generation"](#-generation) section).

## 3. Project structure

The [`pipeline.py`](./pipeline.py) file is the entry point to run the whole pipeline on a folder containing the input image at its root and a `masks/` sub-directory containing all user defined masks. The [`train.py`](./train.py) file is used to train the decomposition model. The most important files are shown here:
```
.
├── capture/        % Module for decomposition
│ ├── callbacks/    % Lightning trainer callbacks
│ ├── data/         % Dataset, subsets, Lightning datamodules
│ ├── render/       % 2D physics based renderer
│ ├── utils/        % Utility functions
│ └── source/       % Network, loss, and LightningModule
│   └── routine.py  % Training loop
│
└── concept/        % Module for inversion and texture generation
  ├── crop.py       % Square crop extraction from image and masks
  ├── invert.py     % Optimization code to learn the concept S*
  └── infer.py      % Inference code to generate texture from S*
```
If you have any questions, post via the [*issues tracker*](https://github.com/astra-vision/MaterialPalette/issues) or contact the corresponding author.

## 4. (optional) Training

We provide the pre-trained decomposition weights (see ["Installation"](#1-installation)). However, if you are looking to retrain the domain adaptive model for your own purposes, we provide the code to do so. Our method relies on the training of a multi-task network on labeled (real) and unlabeled (synthetic) images, *jointly*. In case you wish to retrain on the same datasets, you will have to download both the ***AmbientCG*** and ***TexSD*** datasets.

First download the PBR materials (source) dataset from [AmbientCG](https://ambientcg.com/):
```
python capture/data/download.py path/to/target/directory
```

To run the training script, use:
```
python train.py --config=path/to/yml/config
```

<sup>Additional options can be found with `python train.py --help`.</sup>

> [!NOTE]
> The decomposition model allows estimating the pixel-wise BRDF maps from a single texture image input.

## Acknowledgments
This research project was mainly funded by the French Agence Nationale de la Recherche (ANR) as part of project SIGHT (ANR-20-CE23-0016). Fabio Pizzati was partially funded by KAUST (Grant DFR07910). Results were obtained using HPC resources from GENCI-IDRIS (Grant 2023-AD011014389).

The repository contains code taken from [`PEFT`](https://github.com/huggingface/peft), [`SVBRDF-Estimation`](https://github.com/mworchel/svbrdf-estimation/tree/master), [`DenseMTL`](https://github.com/astra-vision/DenseMTL). As for visualization, we used [`DeepBump`](https://github.com/HugoTini/DeepBump) and [**Blender**](https://www.blender.org/). Credit to Runway for providing us all the [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) model weights. All images and 3D scenes used in this work have permissive licenses. Special credits to [**AmbientCG**](https://ambientcg.com/list) for the huge work.

The authors would also like to thank all members of the [Astra-Vision](https://astra-vision.github.io/) team for their valuable feedback.

## License
If you find this code useful, please cite our paper:
```
@inproceedings{lopes2024material,
    author = {Lopes, Ivan and Pizzati, Fabio and de Charette, Raoul},
    title = {Material Palette: Extraction of Materials from a Single Image},
    booktitle = {CVPR},
    year = {2024},
    project = {https://astra-vision.github.io/MaterialPalette/}
}
```
**Material Palette** is released under [MIT License](./LICENSE).


> [!CAUTION]
> The diffusion checkpoint that is used in the material extraction part of this project is licensed under the CreativeML Open RAIL-M license. Please note that the license includes specific obligations (such as marking modified files) that are not covered by the MIT license of this repository. Users are responsible for reviewing and complying with both licenses when using or modifying the model. See model license for details.

> [!NOTE]
> The specific checkpoint `runwayml/stable-diffusion-v1-5` is no longer available on HuggingFace models. As a user, you may wish to use a more up-to-date version or resort to using a fork of this checkpoint from a 3rd party repository.

---

[🢁 jump to top](#material-palette-extraction-of-materials-from-a-single-image-cvpr-2024)
