#!/bin/bash
docker stop watering_forecasting
docker rm watering_forecasting
docker build -t watering_forecasting .
docker run -u $(id -u):$(id -g) --name watering_forecasting --volume $(pwd):/home/watering_forecasting --detach -t watering_forecasting
docker exec watering_forecasting /bin/bash -c "export PYTHONHASHSEED=42 && python src/run_experiments.py"