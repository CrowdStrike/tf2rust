import argparse
import os
from pathlib import Path

from tensorflow.keras.layers import (
    AlphaDropout,
    BatchNormalization,
    Dropout,
    GaussianNoise,
    SpatialDropout1D,
)

from .nodes import (
    activationNode,
    batchNormalizationNode,
    concatenateNode,
    conv1dNode,
    denseNode,
    embeddingNode,
    flattenNode,
    globalAveragePooling1dNode,
    inputLayerNode,
    maxPooling1dNode,
    multiplyNode,
    reshapeNode,
    tensorFlowAdd2Node,
    tensorFlowMeanNode,
    thresholdedrelu,
    dropoutNode,
)

parser = argparse.ArgumentParser(description="Arguments for the python scripts")
parser.add_argument(
    "--path_to_tf_model",
    type=str,
    default=None,
    help="The path to the TF model intended to converts (only .pb or .h5 format supported)",
)
parser.add_argument(
    "--path_to_save",
    type=str,
    default=None,
    help="The path where to store the conversion.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default=None,
    help="The name of the model to be converted.",
)
parser.add_argument(
    "--binary_classification",
    type=str,
    default="True",
    help="A flag specifying whether the model is a binary classifier or not.",
)
parser.add_argument(
    "--enable_inplace",
    type=str,
    default="True",
    help="Enable inplace operations in model.rs",
)
parser.add_argument(
    "--enable_memdrop", type=str, default="True", help="Enable memory drop in model.rs"
)
parser.add_argument(
    "--path_to_fv",
    type=str,
    default=None,
    help="The path to the fv (npz format) having as keys the names of the InputLayers (e.g. character_level, word_level, extra_level)",
)
args = parser.parse_args()


PATH_TO_TF_MODEL = args.path_to_tf_model
assert PATH_TO_TF_MODEL is not None
PATH_TO_TF_MODEL = Path(PATH_TO_TF_MODEL)

PATH_TO_SAVE = args.path_to_save
assert PATH_TO_SAVE is not None
PATH_TO_SAVE = Path(PATH_TO_SAVE)

MODEL_NAME = args.model_name
assert MODEL_NAME is not None

FILE_PATH_FV = args.path_to_fv


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


ENABLE_INPLACE = str2bool(args.enable_inplace)
ENABLE_MEMDROP = str2bool(args.enable_memdrop)
BINARY_CLASSIFICATION = str2bool(args.binary_classification)

os.makedirs(PATH_TO_SAVE, exist_ok=True)
os.makedirs(PATH_TO_SAVE.joinpath("saved_model_from_tensorflow/"), exist_ok=True)
os.makedirs(PATH_TO_SAVE.joinpath("rust_generated_code/"), exist_ok=True)

PROJECT_PATH = PATH_TO_SAVE.joinpath("rust_generated_code/")

FILE_PATH_WEIGHTS = PATH_TO_SAVE.joinpath("saved_model_from_tensorflow/").joinpath(
    "model_weights.npz"
)
FILE_PATH_MODEL_ARCHITECTURE = PATH_TO_SAVE.joinpath(
    "saved_model_from_tensorflow/"
).joinpath("model_architecture.json")
FILE_PATH_COMPUTATIONAL_GRAPH = PATH_TO_SAVE.joinpath(
    "saved_model_from_tensorflow/"
).joinpath("computation_graph.json")
FILE_PATH_OVERVIEW = PATH_TO_SAVE.joinpath("saved_model_from_tensorflow/").joinpath(
    "model_overview.png"
)

DELETE_LAYERS = (Dropout, SpatialDropout1D, GaussianNoise, AlphaDropout)

LAYERS_DICTIONARY = {
    "InputLayer": inputLayerNode.InputLayerNode,
    "Embedding": embeddingNode.EmbeddingNode,
    "Dense": denseNode.DenseNode,
    "Conv1D": conv1dNode.Conv1DNode,
    "MaxPooling1D": maxPooling1dNode.MaxPool1dNode,
    "Concatenate": concatenateNode.ConcatenateNode,
    "Flatten": flattenNode.FlattenNode,
    "Reshape": reshapeNode.ReshapeNode,
    "Multiply": multiplyNode.MultiplyNode,
    "GlobalAveragePooling1D": globalAveragePooling1dNode.GlobalAveragePooling1DNode,
    "Activation": activationNode.ActivationNode,
    "ThresholdedReLU": thresholdedrelu.ThresholdedReLU,
    "BatchNormalization": batchNormalizationNode.BatchNormalizationNode,
    "TensorFlowOpLayer": {
        "mean": tensorFlowMeanNode.TensorFlowMeanNode,
        "add": tensorFlowAdd2Node.TensorFlowADD2Node,
    },
    "Dropout": dropoutNode.DropoutNode,
}


def get_class(class_name, layer_name):
    layer_name = layer_name.lower()

    target_class = None
    if class_name in LAYERS_DICTIONARY:
        if class_name == "TensorFlowOpLayer":
            if "mean" in layer_name:
                target_class = LAYERS_DICTIONARY[class_name]["mean"]
            elif "add" in layer_name:
                target_class = LAYERS_DICTIONARY[class_name]["add"]
        else:
            target_class = LAYERS_DICTIONARY[class_name]

    if not target_class:
        print("Class_name: {}, layer_name: {}".format(class_name, layer_name))
        raise Exception("Unknown type of layer")

    return target_class
