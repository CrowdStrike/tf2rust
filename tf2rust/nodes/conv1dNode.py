from .node import Node


class Conv1DNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        config = layer_info["config"]
        self.kernel_size = config["kernel_size"][0]
        self.strides = config["strides"][0]
        self.padding = config["padding"]
        self.dilation_rate = config["dilation_rate"][0]
        self.groups = config["groups"]
        self.activation = config["activation"].capitalize()
        self.type = "tensorflow_layers::Conv1DLayer"

    def initialize_layer(self):
        assert len(self.weights_list) >= 1
        assert len(self.weights_list[0].shape) == 3

        padding = self.get_padding(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding_type=self.padding,
        )
        operation_list = []

        for i, weights in enumerate(self.weights_list):
            operation_list.append(
                'let {node.name}_weight_{idx}: Array{dimensions}<{node.dtype}> = weights_dict.by_name("{node.name}_weight_{idx}.npy")?;'.format(
                    node=self, idx=i, dimensions=len(weights.shape)
                )
            )

        args = ", ".join(
            ["{}_weight_{}".format(self.name, i) for i in range(len(self.weights_list))]
        )

        # if bias doesnt exist
        if len(self.weights_list) == 1:
            args = ", ".join(
                [args, "Array1::zeros({})".format(self.weights_list[0].shape[0])]
            )

        layer_declaration = "let {node.name} = {node.type}::new({input}, {node.strides}, vec!{padding}, {node.dilation_rate}, {node.groups}, tensorflow_layers::Activation::{node.activation});".format(
            node=self, input=args, padding=padding
        )

        operation_list.append(layer_declaration)

        return operation_list

    def declare_build(self):
        return (self.name, self.type)

    def apply_layer(self):
        assert len(self.connections["inbounds"]) == 1
        assert (
            len(self.parents_name) == 1
        ), "Node {} has parents {}. It should have exactly one parent".format(
            self.name, self.parents_name
        )

        return [
            "let {declare_mut}out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = self.{node.name}.apply(&out_{input});".format(
                declare_mut=self._format_mut(), node=self, input=self.parents_name[0]
            )
        ]
