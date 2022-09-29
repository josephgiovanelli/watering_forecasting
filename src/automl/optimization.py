# Scikit-learn provides a set of machine learning techniques
import traceback
import numpy as np
import pandas as pd

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
from utils.persistent_system import PersistentSystem

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

conf = 0  # configuration counter


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


def instantiate_pipeline(prototype, seed, config, window_size, y_size):
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

        if config[step]["type"] == "PersistentSystem":
            operator_parameters["window_size"] = window_size
            operator_parameters["y_size"] = y_size
            operator = globals()[config[step]["type"]](**operator_parameters)
        # Instantiate the operator/algorithm, if random_state is in the hyper-parameters
        # of the operator, add it
        elif "random_state" in globals()[config[step]["type"]]().get_params():
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


def scikitlearn_objective(
    X_train, y_train, X_val, y_val, X_test, y_test, window_size, seed, config
):
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
        _type_: the predicted y_train, y_val and y_test
    """
    # Get the prototype from the config
    # (i.e., the order of the pre-processing transformations + the ML algorithm)
    prototype = get_prototype(config)

    # Instantiate the pipeline according to the current config
    # (i.e., at each step we put an operator with specific hyper-parameters)
    pipeline = instantiate_pipeline(
        prototype, seed, config, window_size, y_train.shape[1]
    )

    # Normalization
    (
        X_train,
        y_train,
        y_train_scaler,
        X_val,
        y_val_scaler,
        X_test,
        y_test_scaler,
    ) = normalize_data(X_train, y_train, X_val, y_val, X_test, y_test, window_size)

    # Fit and prediction
    estimator = pipeline.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_val_pred = estimator.predict(X_val)
    y_test_pred = estimator.predict(X_test)

    # Inverse normalization
    y_train_pred, y_val_pred, y_test_pred = denormalize_data(
        y_train_pred,
        y_train_scaler,
        y_val_pred,
        y_val_scaler,
        y_test_pred,
        y_test_scaler,
    )

    return y_train_pred, y_val_pred, y_test_pred


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


def keras_objective(X_train, y_train, X_val, y_val, X_test, statistics, seed, config):
    """Objective function to optimize when Keras NNs are used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        X_test (_type_): the test set
        statistics (_type_): the dictionary containing dataset statistics
        seed (_type_): the seed for reproducibility
        config (_type_): the configuration to explore

    Returns:
        _type_: the predicted y_train, y_val and y_test
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
    y_train_pred = dnn.predict(X_train)
    y_val_pred = dnn.predict(X_val)
    y_test_pred = dnn.predict(X_test)

    return y_train_pred, y_val_pred, y_test_pred


def objective(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    window_size,
    output_horizon,
    statistics,
    seed,
    config,
):
    """Function to optimize (i.e., the order of the pre-processing transformations + the ML algorithm)

    Args:
        X_train (_type_): training data
        y_train (_type_): training data labels (ground truth)
        X_val (_type_): validation data
        y_val (_type_): validation data labels (ground truth)
        X_test (_type_): test data
        y_test (_type_): test data labels (ground truth)
        window_size (_type_): the specified window size
        output_horizon (_type_): the specified output horizon in hours
        statistics (_type_): the dictionary containing dataset statistics
        seed (_type_): seed for reproducibility
        config (_type_): the current config to visit

    Raises:
        Exception: It tells if something went wrong during the optimization

    Returns:
        _type_: dictionary with the result
    """
    global conf

    # Set the result if the config failed to be evaluated
    result = {
        "train_raw_scores": {
            f"train_raw_score_{sensor}": float("-inf")
            for sensor in np.arange(0, output_horizon * 12)
        },
        "val_raw_scores": {
            f"val_raw_score_{sensor}": float("-inf")
            for sensor in np.arange(0, output_horizon * 12)
        },
        "test_raw_scores": {
            f"test_raw_score_{sensor}": float("-inf")
            for sensor in np.arange(0, output_horizon * 12)
        },
        "train_score": float("-inf"),
        "val_score": float("-inf"),
        "test_score": float("-inf"),
        "status": "fail",
    }

    try:
        if config["regression"]["type"] == "keras":
            y_train_pred, y_val_pred, y_test_pred = keras_objective(
                X_train.to_numpy(),
                y_train.to_numpy(),
                X_val.to_numpy(),
                y_val.to_numpy(),
                X_test.to_numpy(),
                statistics,
                seed,
                config,
            )
        else:
            y_train_pred, y_val_pred, y_test_pred = scikitlearn_objective(
                X_train.to_numpy(),
                y_train.to_numpy(),
                X_val.to_numpy(),
                y_val.to_numpy(),
                X_test.to_numpy(),
                y_test.to_numpy(),
                window_size,
                seed,
                config,
            )

        # Export predictions to csv files
        pd.DataFrame(y_train_pred, columns=y_train.columns, index=y_train.index).to_csv(
            "resources/predictions/conf_{}_train.csv".format(conf)
        )
        pd.DataFrame(y_val_pred, columns=y_val.columns, index=y_val.index).to_csv(
            "resources/predictions/conf_{}_val.csv".format(conf)
        )
        pd.DataFrame(y_test_pred, columns=y_test.columns, index=y_test.index).to_csv(
            "resources/predictions/conf_{}_test.csv".format(conf)
        )

        # Compute raw RMSE (a value for each sensor)
        train_raw_rmse = mean_squared_error(
            y_train, y_train_pred, multioutput="raw_values", squared=False
        )
        for idx, rmse in enumerate(train_raw_rmse):
            result["train_raw_scores"][f"train_raw_score_{idx}"] = rmse

        val_raw_rmse = mean_squared_error(
            y_val, y_val_pred, multioutput="raw_values", squared=False
        )
        for idx, rmse in enumerate(val_raw_rmse):
            result["val_raw_scores"][f"val_raw_score_{idx}"] = rmse

        test_raw_rmse = mean_squared_error(
            y_test, y_test_pred, multioutput="raw_values", squared=False
        )
        for idx, rmse in enumerate(test_raw_rmse):
            result["test_raw_scores"][f"test_raw_score_{idx}"] = rmse

        # Compute average RMSE
        result["train_score"] = mean_squared_error(y_train, y_train_pred, squared=False)
        result["val_score"] = mean_squared_error(y_val, y_val_pred, squared=False)
        result["test_score"] = mean_squared_error(y_test, y_test_pred, squared=False)

        # If something is NaN, raise an exception
        for metric in result:
            if metric != "status" and metric != "conf":
                if metric in ["train_raw_scores", "val_raw_scores", "test_raw_scores"]:
                    for elem in result[metric]:
                        if np.isnan(result[metric][elem]):
                            result[metric][elem] = float("-inf")
                            raise Exception(f"The result for {config} was")
                else:
                    if np.isnan(result[metric]):
                        result[metric] = float("-inf")
                        raise Exception(f"The result for {config} was")
        result["status"] = "success"
        result["conf"] = conf

        conf += 1

    except Exception as e:
        print(
            f"""MyException: {e}"""
            #   {traceback.print_exc()}"""
        )

    return result