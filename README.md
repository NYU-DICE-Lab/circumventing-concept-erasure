# Circumventing Concept Erasure Methods For Text-to-Image Models

###  [Project Website](https://nyu-dice-lab.github.io/CCE/) | [Arxiv Preprint](https://arxiv.org/abs/2308.01508) <br>

<div align='center'>
<img src = 'images/headline.png'>
</div>

## Starting Guide
To get started, create a new conda environment and install the required packages:

```bash
conda create -f environment.yml
```

Optional: Get used to the [🤗Textual Inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) GitHub repository.

## Training Guide
We provide Concept Inversion code for the following methods:
- [Erased Stable Diffusion (ESD)](https://github.com/rohitgandikota/erasing) in the `/esd` folder.
- [Forget-Me-Not (FMN)](https://github.com/SHI-Labs/Forget-Me-Not) in the `/fmn` folder.
- [Selective Amnesia (SA)](https://github.com/clear-nus/selective-amnesia) in the `/sa` folder.
- [Negative Prompt (NP)](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt) in the `/np` folder.
- [Safe Latent Diffusion (SLD)](https://github.com/ml-research/safe-latent-diffusion) in the `/sld` folder.

Please refer to each folder for additional instructions, which is designed to be self-contained.

## Acknowledgements
We would like to thank the authors of [🤗Difusers](https://github.com/huggingface/diffusers/)for relesing their helpful codebases.
