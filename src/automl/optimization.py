# Scikit-learn provides a set of machine learning techniques
import traceback
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

## Normalization operators
from sklearn.preprocessing import MinMaxScaler

## Regression algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)  # this estimator is much faster than GradientBoostingRegressor for big datasets (n_samples >= 10 000)
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.svm import SVR

# NNs
import tensorflow as tf
from tensorflow import keras

from utils.data_acquisition import (
    normalize_data,
    denormalize_data,
    normalization,
    inverse_normalization,
)


def get_prototype(config):
    """Create the prototype (i.e., the order of the pre-processing transformations + the ml algorithm)

    Args:
        config (_type_): the current config to visit

    Raises:
        NameError: if the hyper-parameter "prototype" is not in the config

    Returns:
        _type_: a list of string, each string is a step
    """
    ml_pipeline = config["prototype"]
    if ml_pipeline is None:
        raise NameError("No prototype specified")
    else:
        ml_pipeline = ml_pipeline.split("_")
    return ml_pipeline


def instantiate_pipeline(prototype, seed, config):
    """Create the pipeline instantiation from the prototype
    Args:
        prototype (_type_): sequence of pre-processing steps
        seed (_type_): seed for reproducibility
        config (_type_): the current config to visit

    Returns:
        _type_: the pipeline instantiation
    """
    pipeline = []
    # Iterate each step in the prototype
    for step in prototype:
        # Considering the chosen operator of the step at hand,
        # remove the "type" hyper-parameter and build a dict
        # with just the hyper-parameters that operator expects
        operator_parameters = {
            param_name: globals()[config[step][param_name]]()
            if param_name == "base_estimator"  #
            else config[step][param_name]  #
            for param_name in config[step]
            if param_name != "super_type" and param_name != "type"
        }

        if config[step]["type"] == "MLPRegressor":
            operator_parameters["hidden_layer_sizes"] = eval(
                operator_parameters["hidden_layer_sizes"]
            )

        # Instantiate the operator/algorithm, if random_state is in the hyper-parameters
        # of the operator, add it
        if "random_state" in globals()[config[step]["type"]]().get_params():
            if "super_type" in config[step]:
                operator = globals()[config[step]["super_type"]](
                    globals()[config[step]["type"]](
                        random_state=seed, **operator_parameters
                    )
                )
            else:
                operator = globals()[config[step]["type"]](
                    random_state=seed, **operator_parameters
                )
        elif "super_type" in config[step]:
            operator = globals()[config[step]["super_type"]](
                globals()[config[step]["type"]](**operator_parameters)
            )
        else:
            operator = globals()[config[step]["type"]](**operator_parameters)

        # Add the operator to the array
        pipeline.append([step, operator])

    # Instantiate the pipeline from the array
    return Pipeline(pipeline)


def scikitlearn_objective(X_train, y_train, X_val, y_val, window_size, seed, config):
    """Objective function to optimize when Scikit-learn regressors are used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        window_size (_type_): the specified window size
        seed (_type_): the seed for reproducibility
        config (_type_): the configuration to explore

    Returns:
        _type_: the predicted y_val
    """
    # Get the prototype from the config
    # (i.e., the order of the pre-processing transformations + the ML algorithm)
    prototype = get_prototype(config)

    # Instantiate the pipeline according to the current config
    # (i.e., at each step we put an operator with specific hyper-parameters)
    pipeline = instantiate_pipeline(prototype, seed, config)

    # Normalization
    X_train, y_train, X_val, y_val_scaler = normalize_data(
        X_train, y_train, X_val, y_val, window_size
    )

    # Fit and prediction
    estimator = pipeline.fit(X_train, y_train)
    y_val_pred = estimator.predict(X_val)

    # Inverse normalization
    y_val_pred = denormalize_data(y_val_pred, y_val_scaler)

    return y_val_pred


def build_dnn(
    input_count,
    output_count,
    neuron_count_per_hidden_layer,
    activation,
    last_activation,
    dropout,
    statistics,
):
    """Create the neural network architecture.

    Args:
        input_count (_type_): number of input neurons
        output_count (_type_): number of output neurons
        neuron_count_per_hidden_layer (_type_): number of neurons for each hidden layer
        activation (_type_): activation function for the hidden layers
        last_activation (_type_): activation funcion for the output layer
        dropout (_type_): dropout rate
        statistics (_type_): dictionary containing dataset statistics

    Returns:
        _type_: neural network model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_count)))

    # Normalization
    model.add(
        keras.layers.Lambda(normalization, arguments={"norm_parameters": statistics})
    )

    for n in neuron_count_per_hidden_layer:
        model.add(keras.layers.Dense(n, activation=activation))
        # model.add(keras.layers.Dropout(rate=dropout))

    model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(output_count, activation=last_activation))

    # Inverse normalization
    model.add(
        keras.layers.Lambda(
            inverse_normalization, arguments={"norm_parameters": statistics}
        )
    )

    return model


def keras_objective(X_train, y_train, X_val, y_val, statistics, seed, config):
    """Objective function to optimize when Keras NNs are used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        statistics (_type_): the dictionary containing dataset statistics
        seed (_type_): the seed for reproducibility
        config (_type_): the configuration to explore

    Returns:
        _type_: the predicted y_val
    """
    tf.random.set_seed(seed)

    # Create the model
    dnn = build_dnn(
        X_train.shape[1],
        y_train.shape[1],
        eval(config["regression"]["neuron_count_per_hidden_layer"]),
        config["regression"]["activation_function"],
        config["regression"]["last_activation_function"],
        config["regression"]["dropout"],
        statistics,
    )

    # Compile the model
    dnn.compile(
        loss="mse",
        optimizer=config["regression"]["optimizer"],
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    # Fit the model
    patience = 50
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    dnn.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["regression"]["num_epochs"],
        batch_size=config["regression"]["batch_size"],
        shuffle=True,
        callbacks=[early_stop],
    )

    # Prediciton
    y_val_pred = dnn.predict(X_val)

    return y_val_pred


def objective(X_train, y_train, X_val, y_val, window_size, statistics, seed, config):
    """Function to optimize (i.e., the order of the pre-processing transformations + the ML algorithm)

    Args:
        X_train (_type_): training data
        y_train (_type_): training data labels (ground truth)
        X_val (_type_): validation data
        y_val (_type_): validation data labels (ground truth)
        window_size (_type_): the specified window size
        statistics (_type_): the dictionary containing dataset statistics
        seed (_type_): seed for reproducibility
        config (_type_): the current config to visit

    Raises:
        Exception: It tells if something went wrong during the optimization

    Returns:
        _type_: dictionary with the result
    """
    # Set the result if the config failed to be evaluated
    result = {"score": float("-inf"), "status": "fail"}

    try:
        if config["regression"]["type"] == "keras":
            y_val_pred = keras_objective(
                X_train, y_train, X_val, y_val, statistics, seed, config
            )
        else:
            y_val_pred = scikitlearn_objective(
                X_train, y_train, X_val, y_val, window_size, seed, config
            )

        score = mean_squared_error(y_val, y_val_pred, squared=False)

        result["score"] = score

        # If it is NaN, raise an exception
        if np.isnan(result["score"]):
            result["score"] = float("-inf")
            raise Exception(f"The result for {config} was")
        result["status"] = "success"

    except Exception as e:
        print(
            f"""MyException: {e}"""
            #   {traceback.print_exc()}"""
        )

    return result
