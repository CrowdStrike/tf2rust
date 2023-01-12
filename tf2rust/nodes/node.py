class Node:
    def __init__(self, layer_info, layer_weights):
        self.python_to_rust_primitive_mappings = {
            "int16": "usize",
            "int32": "usize",
            "int64": "usize",
            "uint16": "u16",
            "float32": "f32",
            "float64": "f32",
        }
        self.class_name = layer_info["class_name"]
        self.name = layer_info["name"].lower()
        self.input_shape = self.get_shape(layer_info["input_shape"])
        self.output_shape = self.get_shape(layer_info["output_shape"])
        self.input_dimensions = len(self.input_shape)
        self.output_dimensions = len(self.output_shape)
        self.dtype = self.python_to_rust_primitive_mappings[
            layer_info["config"]["dtype"]
        ]
        self.connections = layer_info["connections"]
        self.weights_list = layer_weights
        self.parents_name = self.connections["inbounds"]
        self.output_as_mut = False
        self.inplace_op = False

    # Override this in the corresponding class if this form is not compatible
    @staticmethod
    def get_weights(layer):
        return layer.get_weights()

    def initialize_layer(self):
        pass

    def can_be_done_inplace(self):
        return False

    def memory_drop(self):
        return ["mem::drop(out_{});".format(self.name)]

    def _format_mut(self):
        if self.output_as_mut:
            return "mut "
        return ""

    def declare_build(self):
        return None

    def apply_layer(self):
        pass

    @staticmethod
    def get_padding(input_shape, output_shape, kernel_size, strides, padding_type):
        # remove batch_size
        assert len(input_shape) == len(output_shape)
        no_dims_input = len(input_shape)

        if isinstance(strides, int):
            strides = [strides] * no_dims_input
        elif isinstance(strides, (list, tuple)):
            assert len(strides) == 1
            strides = [strides[0]] * no_dims_input
        # for the batch_size
        strides[0] = 0

        padding = [(0, 0)] * no_dims_input
        if padding_type.lower() == "valid":
            return padding

        elif padding_type.lower() == "same":
            total_padding = max(
                (output_shape[1] - 1) * strides[1] + kernel_size - input_shape[1], 0
            )
            pad_top = total_padding // 2
            pad_bottom = total_padding - pad_top
            padding[1] = (pad_top, pad_bottom)
            return padding

        elif padding_type.lower() == "causal":
            total_padding = max(
                (output_shape[1] - 1) * strides[1] + kernel_size - input_shape[1], 0
            )
            padding[1] = (total_padding, 0)
            return padding

        raise Exception("Unknown type of padding")

    @staticmethod
    def get_shape(shape):
        if isinstance(shape, tuple):
            return shape
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                return tuple(shape[0])
            return tuple(shape)
        raise Exception("Unknown shape")
