from .node import Node


class GlobalAveragePooling1DNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)

    def apply_layer(self):
        assert len(self.connections["inbounds"]) == 1
        assert (
            len(self.parents_name) == 1
        ), "Node {} has parents {}. It should have exactly one parent".format(
            self.name, self.parents_name
        )

        return [
            "let {declare_mut}out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = out_{input}.mean_axis(Axis(1)).unwrap();".format(
                declare_mut=self._format_mut(), node=self, input=self.parents_name[0]
            )
        ]
