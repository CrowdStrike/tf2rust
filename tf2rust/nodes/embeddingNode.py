from .node import Node


class EmbeddingNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        self.type = "tensorflow_layers::EmbeddingLayer"

    def initialize_layer(self):
        assert len(self.weights_list) == 1
        operation_list = []

        operation_list.append(
            'let {node.name}_weight_0: Array{dimensions}<{node.dtype}> = weights_dict.by_name("{node.name}_weight_0.npy")?;'.format(
                node=self, dimensions=len(self.weights_list[0].shape)
            )
        )

        operation_list.append(
            "let {node.name} = {node.type}::new({node.name}_weight_0);".format(
                node=self
            )
        )

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
