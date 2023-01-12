from .node import Node


class ConcatenateNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        self.axis = (
            self.input_dimensions + layer_info["config"]["axis"]
        ) % self.input_dimensions

    def apply_layer(self):
        args = ", ".join(
            ["out_{}".format(layer_name) for layer_name in self.parents_name]
        )

        return [
            "let {declare_mut}out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = concatenate![Axis({node.axis}), {input}];".format(
                declare_mut=self._format_mut(), node=self, input=args
            )
        ]
