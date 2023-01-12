from .node import Node


class TensorFlowMeanNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        self.axis = (
            self.input_dimensions + layer_info["config"]["constants"]["1"]
        ) % self.input_dimensions

    def apply_layer(self):
        assert len(self.connections["inbounds"]) == 1
        assert (
            len(self.parents_name) == 1
        ), "Node {} has parents {}. It should have exactly one parent".format(
            self.name, self.parents_name
        )

        return [
            "let {declare_mut}out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = out_{input}.mean_axis(Axis({node.axis})).unwrap();".format(
                declare_mut=self._format_mut(), node=self, input=self.parents_name[0]
            )
        ]
