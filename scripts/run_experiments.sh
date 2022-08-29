#!/bin/bash
python src/main.py --window_size 4 --output_horizon 1 --batch_size 10 --input_path "resources/automl_input.json" --output_path "resources/automl_output.json" --seed 42
