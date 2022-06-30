#!/bin/bash
python src/main.py --dataset 187 --metric "balanced_accuracy" --mode "max" --batch_size 100 --input_path "resources/automl_input.json" --output_path "resources/automl_output.json" --seed 42
