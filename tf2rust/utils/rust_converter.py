import json
import os
import re
import shutil
from collections import deque

import numpy as np

from tf2rust.constants import (
    BINARY_CLASSIFICATION,
    ENABLE_INPLACE,
    ENABLE_MEMDROP,
    FILE_PATH_COMPUTATIONAL_GRAPH,
    FILE_PATH_FV,
    FILE_PATH_MODEL_ARCHITECTURE,
    FILE_PATH_WEIGHTS,
    MODEL_NAME,
    PROJECT_PATH,
    get_class,
)
from tf2rust.nodes.inputLayerNode import InputLayerNode

MODEL_SERIALIZED_FILE_NAME = "tf_model"


# This implementation assumes that the graph is acyclic as there are will be no valid topological sort otherwise.
def topological_sort(graph):
    def topological_sort_utils(node, visited, stack):
        visited[node] = True

        for neigh in graph[node]["outbounds"]:
            if visited[neigh] is False:
                topological_sort_utils(neigh, visited, stack)

        stack.append(node)

    visited, stack = {node: False for node in graph}, []
    for node in graph:
        if not visited[node]:
            topological_sort_utils(node, visited, stack)

    return stack[::-1]


def get_option_preferences(nodes_dict, computational_graph):
    def _initialize_queue_and_visited():
        in_degree = {
            layer_name: len(computational_graph[layer_name]["inbounds"])
            for layer_name in computational_graph
        }
        queue = deque(
            [layer_name for layer_name in in_degree if in_degree[layer_name] == 0]
        )
        visited = {
            layer_name: True for layer_name in in_degree if in_degree[layer_name] == 0
        }

        return in_degree, visited, queue

    layers_option = {
        layer_name: {
            "can_have_output_mut": True,
            "can_be_done_inplace": nodes_dict[layer_name].can_be_done_inplace(),
        }
        for layer_name in computational_graph
    }
    for layer_name in layers_option:
        if isinstance(nodes_dict[layer_name], InputLayerNode):
            layers_option[layer_name] = {
                "can_have_output_mut": False,
                "can_be_done_inplace": False,
            }

    graph_parents = {
        layer_name: computational_graph[layer_name]["inbounds"]
        for layer_name in computational_graph
    }
    graph_children = {
        layer_name: computational_graph[layer_name]["outbounds"]
        for layer_name in computational_graph
    }

    in_degree_remaining, visited, queue = _initialize_queue_and_visited()
    while len(queue) > 0:
        curr_layer_name = queue.popleft()
        number_of_special_children = sum(
            [
                1
                for child_name in graph_children[curr_layer_name]
                if layers_option[child_name]["can_be_done_inplace"]
            ]
        )
        if (
            len(graph_children[curr_layer_name]) >= 2
            and number_of_special_children >= 1
        ):
            for child_name in graph_children[curr_layer_name]:
                layers_option[child_name]["can_be_done_inplace"] = False

        for child_name in graph_children[curr_layer_name]:
            in_degree_remaining[child_name] -= 1
            if in_degree_remaining[child_name] == 0 and child_name not in visited:
                visited[child_name] = True
                queue.append(child_name)

    # Remove layers that can be done in place, but doesn't have all the inputs set to mut
    # Additionally, compute the output name for each layer (taking into consideration inplace operations).
    in_nodes = {
        layer_name: {
            parent_name: parent_name for parent_name in graph_parents[layer_name]
        }
        for layer_name in graph_parents
    }
    out_nodes = {layer_name: layer_name for layer_name in computational_graph}

    in_degree_remaining, visited, queue = _initialize_queue_and_visited()
    necessary_outputs_mut = {}
    while len(queue) > 0:
        curr_layer_name = queue.popleft()

        if layers_option[curr_layer_name]["can_be_done_inplace"]:
            number_of_parents_with_mut_output = sum(
                [
                    1
                    for direct_parent_name in in_nodes[curr_layer_name]
                    if layers_option[direct_parent_name]["can_have_output_mut"]
                ]
            )

            if number_of_parents_with_mut_output != len(in_nodes[curr_layer_name]):
                layers_option[curr_layer_name]["can_be_done_inplace"] = False
                out_nodes[curr_layer_name] = curr_layer_name
            else:
                assert len(in_nodes[curr_layer_name]) == 1
                [direct_parent_name] = in_nodes[curr_layer_name].keys()
                necessary_outputs_mut[direct_parent_name] = True
                out_nodes[curr_layer_name] = out_nodes[direct_parent_name]

        else:
            out_nodes[curr_layer_name] = curr_layer_name

        for child_name in graph_children[curr_layer_name]:
            in_degree_remaining[child_name] -= 1
            in_nodes[child_name][curr_layer_name] = out_nodes[curr_layer_name]
            if in_degree_remaining[child_name] == 0 and child_name not in visited:
                visited[child_name] = True
                queue.append(child_name)

    in_degree_remaining, visited, queue = _initialize_queue_and_visited()

    # Update nodes_dict, while removing unnecessary can_have_output_mut from layers
    for layer_name in nodes_dict:
        nodes_dict[layer_name].output_as_mut = layer_name in necessary_outputs_mut
        nodes_dict[layer_name].inplace_op = layers_option[layer_name][
            "can_be_done_inplace"
        ]
        nodes_dict[layer_name].parents_name = list(in_nodes[layer_name].values())

    # Transfer all info into layers_option
    for layer_name in layers_option:
        layers_option[layer_name]["in_nodes"] = in_nodes[layer_name]
        layers_option[layer_name]["out_node"] = out_nodes[layer_name]

    return layers_option


def construct_node(layer_info, layer_weights):
    class_name = layer_info["class_name"]
    layer_name = layer_info["name"].lower()

    target_class = get_class(class_name=class_name, layer_name=layer_name)
    return target_class(layer_info=layer_info, layer_weights=layer_weights)


def get_weights_by_name(layer_name, model_weights):
    layers_list, i = [], 0

    while True:
        aux_name = "{}_weight_{}".format(layer_name, i)
        if aux_name not in model_weights:
            break

        layers_list.append(model_weights[aux_name])
        i += 1

    return layers_list


def define_class_name(name):
    res = re.split("[^0-9a-zA-Z]", name)
    res = "".join([word[0:1].upper() + word[1:] for word in res if word != ""])
    if len(res) < 5 or res[len(res) - 5 :] != " Model":
        res += "Model"
    return res


# Recursively convert None to 1
# eg: [[None, 350], [None, 50, 15], [None, 50]] -> [[1, 350], [1, 50, 15], [1, 50]]
def convert_none_to_one(lst):
    if isinstance(lst, list):
        return [convert_none_to_one(l) for l in lst]
    else:
        return 1 if lst is None else lst


def declare_build(traversal_order, nodes_dict, file_path_build, rsrc_path):
    shutil.copy(FILE_PATH_WEIGHTS, rsrc_path)

    with open(file_path_build, "w") as f:
        f.write(
            """\
// DO NOT EDIT! THIS FILE IS AUTOMATICALLY GENERATED!
#[path = "src/model.rs"]
mod model;

use model::{class_name};
use ndarray::{{Array1, Array2, Array3}};
use std::{{error::Error, io::Write}};

#[allow(clippy::similar_names, clippy::too_many_lines)] // suppress clippy's complaints about our autogenerated code
#[rustfmt::skip]
fn serialize_model() -> Result<Vec<u8>, Box<dyn Error>> {{
    let mut weights_dict = ndarray_npy::NpzReader::new(std::fs::File::open("{file_path}")?)?;

    {body}

    let model = {class_name} {{
        {struct_initializer}
    }};

    Ok(model.serialize())
}}

fn main() -> Result<(), Box<dyn Error>> {{
    let path = std::path::Path::new(&std::env::var("OUT_DIR")?).join("{model_serialized_name}");
    let mut file = std::fs::File::create(&path)?;
    file.write_all(&serialize_model()?)?;
    linkin::link_model(path)
}}

/// Use toolchain specific tools to embed model in the build.
/// The default way to do this was to use `include_bytes!` macro which works just fine for any smaller size models,
/// however, since the models have grown (over 100MB) the include bytes take very long time as well as consumes huge amount of memory on build machines.
/// Due to the above a toolchain specific way of embedding the blob in the file is required.
///
/// Linux targets use `ld` to create obj files.
/// Windows MSVC targets use `resource embedding`.
///
/// Both techniques require specific retrieval methods as per `raw_model` function in the lib source.
mod linkin {{

    use std::{{env, error::Error, fs, path::Path}};

    /// On Linux GNU toolchain we can create object file and include in the final binary with linker directives.
    #[cfg(target_os = "linux")]
    pub fn link_model(path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {{
        use std::{{os::unix::fs as unix_fs, process::Command}};

        const LIB_NAME: &str = "{model_serialized_name}_raw";

        let out_dir = env::var("OUT_DIR")?;
        let out_dir = Path::new(&out_dir);

        // symlink the original lib so we get our choice of symbol names
        let orig_model_path = env::current_dir()?.join(path);
        let model_path = out_dir.join(LIB_NAME);
        match fs::remove_file(&model_path) {{
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => panic!("failed to unlink {{}}: {{}}", model_path.display(), e),
        }}
        unix_fs::symlink(&orig_model_path, &model_path).unwrap();

        // use ld turn the model into a .o
        let object_file = out_dir.join(format!("{{}}.o", LIB_NAME));
        let status = Command::new("ld")
            .current_dir(out_dir)
            .arg("-r")
            .arg("-b")
            .arg("binary")
            .arg(LIB_NAME)
            .arg("-o")
            .arg(&object_file)
            .status()?;
        assert!(
            status.success(),
            "ld failed to convert model to object file"
        );

        // create lib from .o
        let status = Command::new("ar")
            .current_dir(out_dir)
            .arg("cr")
            .arg(format!("lib{{}}.a", LIB_NAME))
            .arg(object_file)
            .status()?;
        assert!(status.success(), "ar failed to create lib from object file");

        // technically we should also
        //   println!("cargo:rerun-if-changed={{}}", orig_model_path)
        // so we rerun if the model file changes, but:
        //   a. model file changes include source code changes (each model update
        //      has a new filename), and
        //   b. checking a 150+ MB model for changes is slooooow.
        println!("cargo:rustc-link-lib=static={{}}", LIB_NAME);
        println!("cargo:rustc-link-search=native={{}}", out_dir.display());

        Ok(())
    }}

    /// On Windows we use resource embedding to be able to later retrieve the data via the Windows API `FindResource` and `LoadResource`
    #[cfg(target_os = "windows")]
    pub fn link_model(_path: impl AsRef<Path>) {{
        // For Resource embedding we need to have defined 2 types.
        // As per MSVC docs `NAME_ID_MODEL_FILE` can be any `u16`.
        const NAME_ID_MODEL_FILE: &str = "101";
        // As per MSVC docs `TYPE_ID_BINARY_FILE` has to be an int over 255 value.
        const TYPE_ID_BINARY_FILE: &str = "333";

        let path_emb = Path::new(&env::var("OUT_DIR")?).join("embed.rc");
        let data = format!(
            r#"
#define BINARY_FILE {{}}
#define MODEL_FILE {{}}
MODEL_FILE BINARY_FILE "{model_serialized_name}"
"#,
            TYPE_ID_BINARY_FILE, NAME_ID_MODEL_FILE
        );
        fs::write(&path_emb, data)?;

        // This crate does some Windows MSVC specific black magic...
        // It finds required tools and converts the blob to an embeddable obj and then includes as a resource as per `rc` file.
        embed_resource::compile(path_emb);

        Ok(())
    }}
}}
""".format(
                model_serialized_name=MODEL_SERIALIZED_FILE_NAME,
                class_name=define_class_name(MODEL_NAME),
                file_path=rsrc_path.joinpath(FILE_PATH_WEIGHTS.name).relative_to(
                    PROJECT_PATH
                ),
                body="\n\t".join(
                    i
                    for lst in (
                        nodes_dict[layer_name].initialize_layer()
                        for layer_name in traversal_order
                    )
                    for i in (lst if isinstance(lst, list) else [])
                ).expandtabs(4),
                struct_initializer=",\n\t\t".join(
                    i[0]
                    for i in (
                        nodes_dict[layer_name].declare_build()
                        for layer_name in traversal_order
                    )
                    if i != None
                ).expandtabs(
                    4
                ),  # we want this to fail if len(i) < 1
            )
        )


def declare_model(
    traversal_order,
    nodes_dict,
    computational_graph,
    model_architecture,
    file_path_model,
    enable_inplace=True,
    enable_memory_drop=True,
):
    rust_class_name = define_class_name(MODEL_NAME)
    input_shapes = convert_none_to_one(
        [
            model_architecture[layer]["input_shape"][0]
            for layer in model_architecture
            if model_architecture[layer]["class_name"] == "InputLayer"
        ]
    )

    start_layers_name = [
        name
        for name in computational_graph
        if len(computational_graph[name]["inbounds"]) == 0
    ]
    assert len(input_shapes) == len(start_layers_name)

    final_layers_name = [
        name
        for name in computational_graph
        if len(computational_graph[name]["outbounds"]) == 0
    ]

    input_dict = {
        name: "input_{}".format(i) for i, name in enumerate(start_layers_name)
    }
    output_dict = {
        name: "output_{}".format(i) for i, name in enumerate(final_layers_name)
    }

    out_degree_remaining = {
        layer_name: len(computational_graph[layer_name]["outbounds"])
        for layer_name in computational_graph
    }

    logic_operations = ["let batch_size = (input_0).shape()[0];"]

    # populate parents_name + output_as_mut + inplace_op of each node
    if enable_inplace:
        get_option_preferences(
            nodes_dict=nodes_dict, computational_graph=computational_graph
        )

    for layer_name in traversal_order:
        layer = nodes_dict[layer_name]
        if isinstance(layer, InputLayerNode):
            logic_operations += nodes_dict[layer_name].apply_layer(
                input_name=input_dict[layer_name]
            )
        else:
            logic_operations += nodes_dict[layer_name].apply_layer()

        # implement memory drop as soon as possible
        if enable_memory_drop:
            if not nodes_dict[layer_name].inplace_op:
                for parent in nodes_dict[layer_name].parents_name:
                    out_degree_remaining[parent] -= 1
                    if out_degree_remaining[parent] == 0:
                        logic_operations += nodes_dict[parent].memory_drop()

    args_list = []
    arg_dtype = None
    for layer_name in input_dict:
        name_var = input_dict[layer_name]
        if arg_dtype is None:
            arg_dtype = nodes_dict[layer_name].dtype
        else:
            # make sure the same data type is used across all input arrays
            # if this fails, the generated code needs to change to allow for different input data types
            assert arg_dtype == nodes_dict[layer_name].dtype
        input_dimensions = nodes_dict[layer_name].input_dimensions

        args_list.append(("{}: ".format(name_var), "Array{}".format(input_dimensions)))

    result_types_list, result_var_list = [], []
    for layer_name in output_dict:
        name_var = "out_{}".format(layer_name)
        result_var_list.append(name_var)

        dtype = nodes_dict[layer_name].dtype
        input_dimensions = nodes_dict[layer_name].input_dimensions
        result_types_list.append("Array{}<{}>".format(input_dimensions, dtype))

    # helper to format multiple params as a tuple
    tuplefy = lambda line, items: "({})".format(line) if items > 1 else line

    with open(file_path_model, "w") as f:
        f.write(
            """\
// DO NOT EDIT! THIS FILE IS AUTOMATICALLY GENERATED!
use ndarray::{{concatenate, Array1, Array2, Array3, Array4, Axis}};
use serde::{{Deserialize, Serialize}};
use std::mem;

/// Translated TensorFlow {model} Model
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(clippy::module_name_repetitions)]
pub(crate) struct {class_name} {{
    {members}
}}

impl {class_name} {{
    fn fv_to_arrays(mut fv: Vec<{dtype}>) -> {input_type} {{
        {tuple_builder}
    }}

    /// Predict from a single feature vector.
    ///
    /// # Errors
    ///
    /// Fails if there is an internal error.
    #[allow(dead_code)] // we're imported by build.rs, but the build script doesn't call `predict`
    #[must_use]
    pub(crate) fn predict(&self, fv: Vec<{dtype}>) -> {return_type} {{
        let {tuple_items} = Self::fv_to_arrays(fv);
        self.predict_from_arrays({tuple_ref_items})
    }}

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    #[rustfmt::skip]
    fn predict_from_arrays(&self, {input}) -> {return_type} {{
        {body}

        {result}
    }}

    /// Serialize this model to a byte vector
    #[allow(dead_code)]
    pub(crate) fn serialize(&self) -> Vec<u8> {{
        bincode::serialize(&self).unwrap()
    }}
}}
""".format(
                model=MODEL_NAME,
                class_name=rust_class_name,
                members="\n\t".join(
                    "pub(crate) {item[0]}: {item[1]},".format(item=item)
                    for item in (
                        nodes_dict[layer_name].declare_build()
                        for layer_name in traversal_order
                    )
                    if item != None
                ).expandtabs(4),
                dtype=arg_dtype,
                input=", ".join(
                    "{}{}<{}>".format(n, t, arg_dtype) for (n, t) in args_list
                ),
                input_type=tuplefy(
                    ", ".join("{}<{}>".format(t, arg_dtype) for (_, t) in args_list),
                    len(args_list),
                ),
                tuple_builder=tuplefy(
                    ",\n\t\t\t".join(
                        "{type}::from_shape_vec(({shape}), fv.drain(..{idx}).collect()).unwrap()".format(
                            type=t,
                            shape=", ".join(str(_) for _ in s),
                            idx=np.array(s).prod(),
                        )
                        for ((_, t), s) in zip(args_list, input_shapes)
                    ).expandtabs(4),
                    len(args_list),
                ),
                tuple_items=tuplefy(
                    ", ".join("input_{}".format(i) for i in range(len(args_list))),
                    len(args_list),
                ),
                tuple_ref_items=", ".join(
                    "input_{}".format(i) for i in range(len(args_list))
                ),
                return_type=", ".join(result_types_list),
                body="\n\t\t".join(logic_operations).expandtabs(4),
                result=", ".join(result_var_list),
            )
        )


def declare_cargo_toml(file_path_cargo_toml):
    with open(file_path_cargo_toml, "w") as f:
        f.write(
            """\
[package]
name = "predictor-example"
version = "0.0.1"
authors = ["Crowdstrike DSCI <dsci-oss@crowdstrike.com>"]
edition = "2021"
description = "Example Predictor"
license = "MIT"
include = ["build.rs", "Cargo.toml", "benches/*", "model/*", "src/*"]

build = "build.rs"

[build-dependencies]
bincode = "1.3.1"
ndarray = { version = "0.15.5" }
ndarray-npy = "0.8.1"
serde = { version = "1.0.188", features = ["derive"] }
tensorflow_layers = { package = "tf-layers", version = "0.4.0" }

[target.'cfg(all(windows, target_env = "msvc"))'.build-dependencies]
embed-resource = "2.3"

[dependencies]
bincode = "1.3.1"
ndarray = { version = "0.15.5", features = ["serde-1"] }
once_cell = "1.18"
serde = { version = "1.0.188", features = ["derive"] }
serde_json = "1.0.107"
tensorflow_layers = { package = "tf-layers", version = "0.4.0" }

[target.'cfg(windows)'.dependencies]
windows = { version = "0.51.1", features = [
    "Win32_System_LibraryLoader",
    "Win32_Foundation",
] }

[dev-dependencies]
base64 = "0.21.4"
criterion = "0.5"
itertools = "0.11.0"
ndarray-npy = "0.8.1"
once_cell = "1.18"

[[bench]]
name = "benchmarks"
harness = false
"""
        )


def declare_lib(file_path_lib):

    if BINARY_CLASSIFICATION:
        predictor_base = "pub use predictor_base::{BinaryClassThresholds, BinaryModelResult, ScanPrediction};"
        return_type = "BinaryModelResult"
        content_predict_from = """\
    let decision_value = MODEL.predict(fv)[[0, 1]];

    BinaryModelResult {
        dirty: ScanPrediction {
            fv_index: 0,
            confidence: THRESHOLDS.dirty_confidence(decision_value),
            decision_value,
        },
    }"""

        test_content = """\
#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;
    use ndarray::{Array1, Array2, Axis};
    use ndarray_npy::NpzReader;
    use std::fs::File;

    #[test]
    fn test_predictions() {
        let (features, expected_predictions): (Array2<usize>, Array1<f32>) = {
            let mut npz = NpzReader::new(File::open("testdata/features.npz").unwrap()).unwrap();

            let fv: Array2<i32> = npz.by_name("inputs.npy").unwrap();
            let predictions: Array1<f32> = npz.by_name("predictions.npy").unwrap();
            (fv.mapv(|elem| elem as usize), predictions)
        };

        for (fv, expected_dirty_score) in
            izip!(features.axis_iter(Axis(0)), expected_predictions.iter())
        {
            let fv_to_vec = fv.into_owned().into_raw_vec();
            let result = predict_from(fv_to_vec);
            let dirty_score = result.dirty.decision_value;
            let tolerance: f32 = 1.0e-5;
            assert!(
                (dirty_score - expected_dirty_score).abs() < tolerance,
                "predicted: {} whereas: {} was expected",
                dirty_score,
                expected_dirty_score
            );
        }
    }
}"""

    else:
        predictor_base = "pub use predictor_base::{BinaryClassThresholds};"
        return_type = "Array1<f32>"
        content_predict_from = """\
    MODEL.predict(fv).index_axis(Axis(0), 0).to_owned()"""
        test_content = ""

    rust_class_name = define_class_name(MODEL_NAME)
    with open(file_path_lib, "w") as f:
        f.write(
            """\
//! Crate for running predictions against the `{class_name}`

use ndarray::{{Array1, Axis}};
use once_cell::sync::Lazy;
{predictor_base}

#[rustfmt::skip]
mod model;
use model::{class_name};

/// Model version.
pub const MODEL_VERSION: u32 = 1;

static MODEL: Lazy<{class_name}> = Lazy::new(|| bincode::deserialize(raw_model::raw_model()).unwrap());
static THRESHOLDS: Lazy<BinaryClassThresholds> =
    Lazy::new(|| serde_json::from_str(include_str!("../model/thresholds.json")).unwrap());

/// Predict from a set of feature vectors.
#[must_use]
pub fn predict_from(fv: Vec<usize>) -> {return_type} {{
{content_predict_from}
}}

/// This module contains a function to load the model from current binary.
mod raw_model {{

    /// On Linux GNU read symbols from the lib and load them.
    #[cfg(target_os = "linux")]
    #[must_use]
    pub(crate) fn raw_model() -> &'static [u8] {{
        use std::ptr::addr_of;

        // On linux, the model is created from an object file and linked in; we
        // have to do some pointer shenanigans to find it. Our build script names
        // the input `predictor_pe_model_raw`, then the `ld` invocation prepends
        // `_binary` and creates a start and end symbol giving us the range.
        extern "C" {{
            static _binary_tf_model_raw_start: u8;
            static _binary_tf_model_raw_end: u8;
        }}

        unsafe {{
            let start = addr_of!(_binary_tf_model_raw_start);
            let end = addr_of!(_binary_tf_model_raw_end);
            let length = (end as usize) - (start as usize);
            std::slice::from_raw_parts(start, length)
        }}
    }}

    /// On Windows MSVC we use `FindResource` and `LoadResource` to get the model from the binary.
    #[cfg(target_os = "windows")]
    #[must_use]
    pub(crate) fn raw_model() -> &'static [u8] {{
        use windows::core::PCWSTR;

        // For Resource embedding we need to have defined 2 types.
        // As per MSVC docs `NAME_ID_MODEL_FILE` can be any `u16`.
        const NAME_ID_MODEL_FILE: usize = 101;

        // As per MSVC docs `TYPE_ID_BINARY_FILE` has to be an int over 255 value.
        const TYPE_ID_BINARY_FILE: usize = 333;

        // We have hardcoded the `MODEL_FILE` name id in the `rc` file, retrieving it via the ID.
        let name_id = PCWSTR(unsafe {{
            // SAFETY: The resource name id is a u16 which is supposed to be constructed with `MAKEINTRESOURCE`
            //         to generate a "string pointer" (the winapi knows its not a pointer)
            core::mem::transmute(NAME_ID_MODEL_FILE)
        }});

        // We have hardcoded the `BINARY_FILE` type id in the `rc` file, retrieving it via the ID.
        let type_id = PCWSTR(unsafe {{
            // SAFETY: The resource type id is a u16 which is supposed to be constructed with `MAKEINTRESOURCE`
            //         to generate a "string pointer" (the winapi knows its not a pointer)
            core::mem::transmute(TYPE_ID_BINARY_FILE)
        }});

        let p: *const u16 = std::ptr::null();
        unsafe {{
            // We need current module handle for the next call
            let handle =
                windows::Win32::System::LibraryLoader::GetModuleHandleW(PCWSTR(p)).unwrap();
            assert!(
                !handle.is_invalid(),
                "Failed to GetModuleHandleW : {{:?}}",
                handle
            );

            // Use `FindResource` to retrieve the pointer to the embedded resource and then load to memory.
            let resource =
                windows::Win32::System::LibraryLoader::FindResourceW(handle, name_id, type_id);
            assert!(
                !resource.is_invalid(),
                "Failed to find resource in PE : {{:?}}",
                resource
            );

            let resource_data =
                windows::Win32::System::LibraryLoader::LoadResource(handle, resource);
            assert!(resource_data != 0, "Failed to load resource");

            let size = windows::Win32::System::LibraryLoader::SizeofResource(handle, resource);

            // Finally get the data from the module.
            let data = windows::Win32::System::LibraryLoader::LockResource(resource_data);
            assert!(!data.is_null(), "Failed to lock resource");

            std::slice::from_raw_parts(data.cast::<u8>(), size as usize)
        }}
    }}
}}

{test_content}
""".format(
                predictor_base=predictor_base,
                class_name=rust_class_name,
                return_type=return_type,
                content_predict_from=content_predict_from,
                test_content=test_content,
            )
        )


def declare_thresholds(file_path_thresholds):
    with open(file_path_thresholds, "w") as f:
        f.write(
            """\
{
  "dirty": {
    "bottom": 0.0,
    "low": 0.0,
    "medium": 0.0,
    "high": 0.0
  }
}
    """
        )


def declare_bench(file_path_bench):
    with open(file_path_bench, "w") as f:
        f.write(
            """\
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Axis, Slice};
use ndarray_npy::NpzReader;
use once_cell::sync::Lazy;
use predictor_example::predict_from;
use std::fs::File;
use std::time::{Duration, Instant};

static FEATURES: Lazy<Array2<usize>> = Lazy::new(|| {
    let mut npz = NpzReader::new(File::open("testdata/features.npz").unwrap()).unwrap();
    let fv: Array2<i32> = npz.by_name("inputs.npy").unwrap();
    fv.mapv(|elem| elem as usize)
});

fn bench_predict_one_sample(c: &mut Criterion) {
    let fv = FEATURES.index_axis(Axis(0), 0).to_vec();
    c.bench_function("Inference for one sample", move |b| {
        b.iter_custom(|iters| {
            let mut duration: Vec<Duration> = Vec::with_capacity(iters as usize);
            for _i in 0..iters {
                let f = fv.clone();

                let start = Instant::now();
                let _ = predict_from(black_box(f));
                let end = start.elapsed();

                duration.push(end)
            }
            duration.iter().sum()
        })
    });
}

fn bench_predict_multiple_samples(c: &mut Criterion) {
    let head_features = FEATURES.slice_axis(Axis(0), Slice::from(0..10));
    c.bench_function("Run inference for a batch of samples", |b| {
        b.iter_custom(|iters| {
            let mut duration: Vec<Duration> = Vec::with_capacity(iters as usize);
            for _i in 0..iters {
                duration.push(
                    head_features
                        .axis_iter(Axis(0))
                        .map(|fv| {
                            let start = Instant::now();
                            let fv_to_vec = fv.into_owned().into_raw_vec();
                            let _ = predict_from(fv_to_vec);
                            start.elapsed()
                        })
                        .sum(),
                );
            }
            duration.iter().sum()
        });
    });
}

criterion_group!(
    benches,
    bench_predict_one_sample,
    bench_predict_multiple_samples
);

criterion_main!(benches);
"""
        )


def prepare_fv(model_architecture, path_feature_vectors, file_path_testdata):
    order_keys = [
        layer_name
        for layer_name in model_architecture
        if model_architecture[layer_name]["class_name"].lower() == "InputLayer".lower()
    ]
    dictionary_arrays = np.load(path_feature_vectors)
    arrays_list = [dictionary_arrays[key] for key in order_keys]

    array_1d = np.concatenate(
        [arr.reshape((-1, np.prod(arr.shape[1:]))) for arr in arrays_list], axis=1
    )
    array_1d = array_1d.astype(np.int32)

    predictions = dictionary_arrays["predictions"]
    predictions = predictions.astype(np.float32)

    save_args = {"inputs": array_1d, "predictions": predictions}
    np.savez(file_path_testdata, **save_args)


def convert_to_rust():
    PROJECT_SRC = PROJECT_PATH.joinpath("src")
    os.makedirs(PROJECT_SRC, exist_ok=True)
    PROJECT_RSRC_PATH = PROJECT_PATH.joinpath("model")
    os.makedirs(PROJECT_RSRC_PATH, exist_ok=True)
    PROJECT_CONFIG_PATH = PROJECT_PATH.joinpath(".cargo")
    os.makedirs(PROJECT_CONFIG_PATH, exist_ok=True)
    PROJECT_TESTDATA_PATH = PROJECT_PATH.joinpath("testdata")
    os.makedirs(PROJECT_TESTDATA_PATH, exist_ok=True)
    PROJECT_BENCH_PATH = PROJECT_PATH.joinpath("benches")
    os.makedirs(PROJECT_BENCH_PATH, exist_ok=True)

    FILE_PATH_MODEL = PROJECT_SRC.joinpath("model.rs")
    FILE_PATH_LIB = PROJECT_SRC.joinpath("lib.rs")
    FILE_PATH_BUILD = PROJECT_PATH.joinpath("build.rs")
    FILE_PATH_CARGO_TOML = PROJECT_PATH.joinpath("Cargo.toml")
    FILE_PATH_THRESHOLDS = PROJECT_RSRC_PATH.joinpath("thresholds.json")
    FILE_PATH_TESTDATA = PROJECT_TESTDATA_PATH.joinpath("features.npz")
    FILE_PATH_BENCH = PROJECT_BENCH_PATH.joinpath("benchmarks.rs")

    model_architecture = json.load(open(FILE_PATH_MODEL_ARCHITECTURE, "r"))
    computational_graph = json.load(open(FILE_PATH_COMPUTATIONAL_GRAPH, "r"))
    model_weights = np.load(FILE_PATH_WEIGHTS, allow_pickle=True)

    traversal_order = topological_sort(computational_graph)

    nodes_dict = {
        layer_name: construct_node(
            layer_info=model_architecture[layer_name],
            layer_weights=get_weights_by_name(layer_name, model_weights),
        )
        for layer_name in computational_graph
    }

    declare_build(
        traversal_order=traversal_order,
        nodes_dict=nodes_dict,
        file_path_build=FILE_PATH_BUILD,
        rsrc_path=PROJECT_RSRC_PATH,
    )
    declare_model(
        traversal_order=traversal_order,
        nodes_dict=nodes_dict,
        computational_graph=computational_graph,
        model_architecture=model_architecture,
        file_path_model=FILE_PATH_MODEL,
        enable_inplace=ENABLE_INPLACE,
        enable_memory_drop=ENABLE_MEMDROP,
    )
    declare_lib(file_path_lib=FILE_PATH_LIB)
    declare_cargo_toml(file_path_cargo_toml=FILE_PATH_CARGO_TOML)
    declare_thresholds(file_path_thresholds=FILE_PATH_THRESHOLDS)
    declare_bench(file_path_bench=FILE_PATH_BENCH)

    if FILE_PATH_FV is not None:
        prepare_fv(
            model_architecture=model_architecture,
            path_feature_vectors=FILE_PATH_FV,
            file_path_testdata=FILE_PATH_TESTDATA,
        )

    print("#### The model was successfully converted into pure Rust code! ####")


if __name__ == "__main__":
    convert_to_rust()
