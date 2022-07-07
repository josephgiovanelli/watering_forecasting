# Structure
- ```.devcontainer```: configuration to instantiate a dev container w/ vscode
- ```data```: raw agro data for the analysis
- ```resources```: example of the automl input (i.e., space) and output (i.e., visited confs)
- ```scripts```: starting point for reproducibility
- ```src```: source code

# Reproducibility

A Docker file is present to build the needed container (virtual envs are kind of "vintage" nowadays).
There are two options:
- [JUST RUN] run either ```scripts/start.sh``` (unix) or ```scripts/start.bat``` (windows) to build the container and launch ```scripts/run_experiments.sh```, which already contains a configuration for the file ```src/main.py```. If you want to run something different to the ```src/main.py```, you should either: (i) modify ```scripts/run_experiments.sh``` to do that or (ii) once the container is running, run ```docker exec watering_forecasting [your command]```, in this case ```watering_forecasting``` is the name of the container and ```[your command]``` could be something like ```python file_name.py --param_name param_value``` or ```bash another_script.sh```. This option is usually use in deployment, not in dev.
- [RUN & DEBUG] open vscode, which should suggest you to open the project in the devcontainer. Here, you can both run and debug each file through the vscode interface. In the container, I installed some plugins that both help to maintain a "good-quality" code (e.g., black formatter) and are useful to develop/see results (csv reader). (To notice: if you open a terminal, you will find yourself inside the container, it is like your world is just the container.)

# Notes

The entry point is ```src/main.py```, which optimizes the objective in ```src/automl/optimization.py```.

## AutoML

The "toy example" used for the space involves two types of pre-processing transformations (i.e., normalization, features engineering) and a bunch of ML algorithms. The implementation are those of scikit-learn. To start, we can enable the usage of NNs with Keras.

### Space loading

In ```src/automl/space_loading.py```, there is a function that converts a space in json in a space suitable for flaml (it is not important to understand it now, it is a mere translation, indeed I didn't spend much time to comment it).

## Utils

We will discuss just data acquisition, since ```src/utils/argparse.py``` and ```arc/utils/json_to_csv.py``` are just scripts of mere translation.

### Data acquisition

In ```src/utils/data_acquisition.py```, there is a function that loads the raw data and provides a dataframe. Something similar should be done when loading raw data from the database.
You can find also the function that loads datasets from openml (it is just to let this example work, but it will be discarded).

## Future
### Rolling window

The following link reports exactly what we need to do.
<https://www.mathworks.com/help/econ/rolling-window-estimation-of-state-space-models.html>

We did not talk about neither *m* (i.e., the size of the rolling window) nor *h* (i.e., the forecasting horizon). For now, just keep them parametrizable. I think we will need an aggregation before applying the rolling window (from hourly to daily data), and then the rolling window will be fine as it is.
