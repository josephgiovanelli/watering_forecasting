#!/bin/bash
python src/main.py --window_size 4 --stride 1 --output_horizon 1 --batch_size 1 --input_path "resources/automl_input.json" --output_path "resources/automl_output.json" --seed 42 --db_address "" --db_port 5432 --db_user "" --db_password ""
