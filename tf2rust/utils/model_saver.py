import json
import os

import numpy as np

# For some obscure reason and maybe conflict with version of `sk-learn` and `tf`,
# `sk-learn` has to be imported before `tf` is ever used... https://github.com/scikit-learn/scikit-learn/issues/14485
import sklearn
import tensorflow as tf
from tensorflow.keras.models import load_model

from tf2rust.constants import (
    DELETE_LAYERS,
    FILE_PATH_COMPUTATIONAL_GRAPH,
    FILE_PATH_MODEL_ARCHITECTURE,
    FILE_PATH_OVERVIEW,
    FILE_PATH_WEIGHTS,
    PATH_TO_TF_MODEL,
    get_class,
)

from .scoring_metrics import tnr, tpr
from .surgeon.operations import delete_layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def sanitize(model):
    initial_number_of_layers = len(model.layers)
    layers_to_delete = [
        layer for layer in model.layers if isinstance(layer, DELETE_LAYERS)
    ]

    for layer_to_delete in layers_to_delete:
        model = delete_layer(model, layer_to_delete)

    tf.keras.utils.plot_model(
        model,
        to_file=FILE_PATH_OVERVIEW,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    print(
        "#### The number of layers before/after sanitizing the model: {}/{} ####".format(
            initial_number_of_layers, len(model.layers)
        )
    )
    return model


def save_rust_model(
    path_to_tf_model,
    file_path_weights,
    file_path_model_architecture,
    file_path_computational_graph,
):
    model = load_model(
        filepath=path_to_tf_model,
        compile=False,
        custom_objects={"tpr": tpr, "tnr": tnr},
    )
    model = sanitize(model)

    # save computational graph
    computational_graph = {}

    json_model = json.loads(model.to_json())["config"]["layers"]
    for layer_dict in json_model:
        # must be lower case
        layer_name = layer_dict["name"].lower()

        if layer_name not in computational_graph:
            computational_graph[layer_name] = {"inbounds": [], "outbounds": []}

        if len(layer_dict["inbound_nodes"]) > 0:
            for in_node in layer_dict["inbound_nodes"][0]:
                inbound_node = in_node[0].lower()
                if inbound_node not in computational_graph:
                    computational_graph[inbound_node] = {
                        "inbounds": [],
                        "outbounds": [],
                    }

                computational_graph[inbound_node]["outbounds"].append(layer_name)
                computational_graph[layer_name]["inbounds"].append(inbound_node)

    with open(file_path_computational_graph, "w+") as json_file:
        json.dump(computational_graph, json_file, indent=4)

    # save architecture + save weights
    dictionary_architecture = {
        layer_dict["name"].lower(): layer_dict for layer_dict in json_model
    }

    for layer in model.layers:
        layer_name = layer.name.lower()
        dictionary_architecture[layer_name]["input_shape"] = tuple(layer.input_shape)
        dictionary_architecture[layer_name]["output_shape"] = tuple(layer.output_shape)
        dictionary_architecture[layer_name]["connections"] = computational_graph[
            layer_name
        ]
        del dictionary_architecture[layer_name]["inbound_nodes"]

    dictionary_weights = {}
    for layer in model.layers:
        layer_name = layer.name.lower()
        class_name = dictionary_architecture[layer_name]["class_name"]
        weights_list = get_class(
            class_name=class_name, layer_name=layer_name
        ).get_weights(layer)

        channels_last_case = (
            "data_format" in dictionary_architecture[layer_name]["config"]
            and dictionary_architecture[layer_name]["config"]["data_format"]
            == "channels_last"
        )
        if channels_last_case:
            dictionary_architecture[layer_name]["config"][
                "data_format"
            ] = "channels_first"

        for i, weight in enumerate(weights_list):
            name_weight = ("{}_weight_{}".format(layer_name, i)).lower()
            if channels_last_case:
                dimensions = len(weight.shape)
                transpose_dimensions = tuple(
                    [dimensions - 1] + [i for i in range(dimensions - 1)]
                )
                dictionary_weights[name_weight] = weight.astype(np.float32).transpose(
                    transpose_dimensions
                )
            else:
                dictionary_weights[name_weight] = weight.astype(np.float32)

    np.savez(file_path_weights, **dictionary_weights)

    with open(file_path_model_architecture, "w+") as json_file:
        json.dump(dictionary_architecture, json_file, indent=4)


def save_tf_model():
    save_rust_model(
        path_to_tf_model=PATH_TO_TF_MODEL,
        file_path_weights=FILE_PATH_WEIGHTS,
        file_path_model_architecture=FILE_PATH_MODEL_ARCHITECTURE,
        file_path_computational_graph=FILE_PATH_COMPUTATIONAL_GRAPH,
    )
    print(
        "#### The model was successfully saved in a suitable format for the Rust converter! ####"
    )


if __name__ == "__main__":
    save_tf_model()
