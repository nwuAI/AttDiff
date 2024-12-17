# AttDiff
##One-step Attribute-control Diffusion Model for Image Inpainting

* **SAIL (Statistic Analysis And Intelligent Learning) Lab of NWU**

* We provide related codes and configuration files to reproduce the "One-step Attribute-control Diffusion Model for Image Inpainting"


Example images

<p align="center">
  <img src="./img/case.png" alt="Image">
</p>


##Introduction
we propose a One-step Attribute-control Diffusion Model for Image Inpainting (AttDiff). AttDiff comprises two key modules: the Attribute random Selection and Edit Module (ASEM) and the Adaptive Fusion Module (AFM).
Unlike previous DM-based methods, AttDiff is as a one-step diffusion model, significantly enhancing restoration speed.

<p align="center">
  <img src="./img/network.png" alt="Image">
</p>


## Train the model
Train with options from a config file:
```bash
python train.py --config configs/celebahq.yml
```


## Inference Dataset
Before running the following commands make sure to put the downloaded weights file into the `pretrained` folder.
```bash
python inference.py
```

## Inference on a Single Image
Before running the following commands make sure to put the downloaded weights file into the `pretrained` folder.
```bash
python inference_single.py
```

## Requirements
  + python3
  + pytorch
  + torchvision
  + numpy
  + Pillow
  + tensorboard
  + pyyaml