# Concept Inversion for Forget-Me-Not

## Setup
First, obtain the pre-trained weights from [Forget-Me-Not](https://github.com/SHI-Labs/Forget-Me-Not) official GitHub repository. If the weights are not included, you will need to train your own models. The final model will be saved in the original Stable Diffusion format. 

Second, to convert it to diffusers format, use the script `convert_original_stable_diffusion_to_diffusers.py` with the flag `--extract_ema`.

## Train
