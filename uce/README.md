# Concept Inversion for Erased Stable Diffusion (ESD)

## Setup
First, obtain the pre-trained weights from [Erased Stable Diffusion](https://erasing.baulab.info/) official GitHub repository. If the weights are not included, you will need to train your own models. 

## Generating Training Images

```bash
python3 generate_images.py --output_dir $OUTPUT_DIR --prompt $PROMPT --mode train --num_train_images $NUM_TRAIN_IMAGES
```

## Train
You can start training by running the script `train.sh`. The current script is set up to perform concept inversion on style. Please change the argument for `--learnable_property`, `--placeholder_token`, and `--initializer_token` accordingly if you want to learn a different property. The supported properties are style, object, and person. Since most of ESD pre-trained weights are finetuned from Stable Diffusion 1.4, you might want to keep `MODEL_NAME` unchanged.

### I2P
To replicate the I2P experiment, first generate the I2P images by running:

```bash
python3 generate_i2p.py --output_dir $OUTPUT_DIR --model_path $MODEL_PATH
```

The output folder will contain a `train` folder that contains the images, and a `metadata.json` file that contains a list of filenames and corresponding prompts. There is also the `--esd_checkpoint` argument if you want to generate images using the ESD models.

For Concept Inversion, you can use the `train.sh` script you but add the `--i2p` flag and specify the path to the `metadata.json` file in the argument `--i2p_metadata_path`.

## Inference
```bash
python3 generate_images.py --output_dir $OUTPUT_DIR --prompt $PROMPT --mode test --seed $SEED
```