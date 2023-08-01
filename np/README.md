# Concept Inversion for Negative Prompt (NP)

## Train
You can start training by running the script `train.sh`. The current script is set up to perform concept inversion on style. Please change the argument for `--learnable_property`, `--placeholder_token`, and `--initializer_token` accordingly if you want to learn a different property. The supported properties are style, object, and person. Additionally, change `MODEL_NAME` to the path of the FMN models.

### I2P
To replicate the I2P experiment, first generate the I2P images by running:

```bash
python3 generate_i2p.py --output_dir $OUTPUT_DIR --model_path $MODEL_PATH
```

The output folder will contain a `train` folder that contains the images, and a `metadata.json` file that contains a list of filenames and corresponding prompts.

For Concept Inversion, you can use the `train.sh` script you but add the `--i2p` flag and specify the path to the `metadata.json` file in the argument `--i2p_metadata_path`.

## Inference
To generate images, use `generate_images.py`.