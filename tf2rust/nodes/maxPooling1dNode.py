from .node import Node


class MaxPool1dNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        config = layer_info["config"]
        self.pool_size = config["pool_size"][0]
        self.strides = config["strides"][0]
        self.padding = config["padding"]
        self.type = "tensorflow_layers::MaxPooling1DLayer"

    def initialize_layer(self):
        padding = self.get_padding(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            kernel_size=self.pool_size,
            strides=self.strides,
            padding_type=self.padding,
        )

        operation_list = [
            "let {node.name} = {node.type}::new({node.pool_size}, {node.strides}, vec!{padding});".format(
                node=self, padding=padding
            )
        ]

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
