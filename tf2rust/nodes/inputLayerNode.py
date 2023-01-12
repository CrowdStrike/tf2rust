from .node import Node


class InputLayerNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)

    def apply_layer(self, input_name):
        assert (
            len(self.connections["inbounds"]) == 0
        ), "Node {} has parents {}. It should have no parents".format(
            self.name, self.parents_name
        )

        return [
            "let out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = {input};".format(
                node=self, input=input_name
            )
        ]
