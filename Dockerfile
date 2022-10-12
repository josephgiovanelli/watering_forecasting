FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
ARG DOCKER_USER=mmeluzzi
RUN addgroup -S $DOCKER_USER && adduser -S $DOCKER_USER -G $DOCKER_USER
USER $DOCKER_USER
RUN apt-get update && apt-get install -y git --no-install-recommends
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt
COPY . /home/watering_forecasting
WORKDIR /home/watering_forecasting
