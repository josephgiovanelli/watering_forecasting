#!/bin/bash
export UID=$(id -u)
export GID=$(id -g)
docker stop watering_forecasting
docker rm watering_forecasting
docker build -t watering_forecasting .
docker run --user $UID:$GID --name watering_forecasting --volume $(pwd):/home/watering_forecasting --detach -t watering_forecasting
docker exec watering_forecasting python src/run_experiments.py