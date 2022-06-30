# Scikit-learn provides a set of machine learning techniques
import traceback
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

## Normalization operators
from sklearn.preprocessing import (
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
)

## Classification algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


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
            param_name: config[step][param_name]
            for param_name in config[step]
            if param_name != "type"
        }
        # The MLPClassifier expects a tuple that specifies the number of layers
        # and the numbers of neurons per layer
        # In json it is not possible to put tuples (we specified two different hyper-parameters)
        if config[step]["type"] == "MLPClassifier":
            operator_parameters["hidden_layer_sizes"] = (
                operator_parameters["n_neurons"]
            ) * operator_parameters["n_hidden_layers"]
            operator_parameters.pop("n_neurons", None)
            operator_parameters.pop("n_hidden_layers", None)

        # Instantiate the operator/algorithm, if random_state is in the hyper-parameters
        # of the operator, add it
        if "random_state" in globals()[config[step]["type"]]().get_params():
            operator = globals()[config[step]["type"]](
                random_state=seed, **operator_parameters
            )
        else:
            operator = globals()[config[step]["type"]](**operator_parameters)

        # Add the operator to the array
        pipeline.append([step, operator])

    # Instantiate the pipeline from the array
    return Pipeline(pipeline)


def objective(X, y, metric, seed, config):
    """Function to optimize (i.e., the order of the pre-processing transformations + the ml algorithm)

    Args:
        X (_type_): data matrix
        y (_type_): data labels (groundt ruth)
        metric (_type_): the metric to optimize
        seed (_type_): seed for reproducibility
        config (_type_): the current config to visit

    Raises:
        Exception: It tells if something went wrong during the optimization

    Returns:
        _type_: dictionary with the result
    """
    # Set the result if the config failed to be evaluated
    result = {metric: float("-inf"), "status": "fail"}

    try:
        # Get the prototype from the config
        # (i.e., the order of the pre-processing transformations + the ml algorithm)
        prototype = get_prototype(config)

        # Instantiate the pipeline according to the current config
        # (i.e., at each step we put an operator with specific hyper-parameters)
        pipeline = instantiate_pipeline(prototype, seed, config)

        # Evaluate it through 10-times cross-validation
        scores = cross_validate(
            pipeline,
            X.copy(),
            y.copy(),
            scoring=[metric],
            cv=10,
            return_estimator=False,
            return_train_score=False,
            verbose=0,
        )

        # Get the metric value
        result[metric] = np.mean(scores["test_" + metric])

        # If it is NaN, raise an exception
        if np.isnan(result[metric]):
            result[metric] = float("-inf")
            raise Exception(f"The result for {config} was")
        result["status"] = "success"

    except Exception as e:
        print(
            f"""MyException: {e}"""
            #   {traceback.print_exc()}"""
        )

    return result
