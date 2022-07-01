# Structure
- ```.devcontainer```: configuration to instantiate a dev container w/ vscode
- ```data```: raw agro data for the analysis
- ```resources```: example of the automl input (i.e., space) and output (i.e., visited confs)
- ```scripts```: starting point for reproducibility
- ```src```: source code

# Reproducibility

A Docker file is present to build the needed container.
There are two options:
- [JUST RUN] run either ```scripts/start.sh``` (unix) or ```scripts/start.bat``` (windows) to build the container and launch ```scripts/run_experiments.sh```, which already contains a configuration;
- [RUN & DEBUG] open vscode, which should suggest you to open the project in the devcontainer.

# Notes

## Data acquisition

In ```src/automl/data_acquisition.py```, there is a function that loads the raw data and provides a dataframe.

## Rolling window

