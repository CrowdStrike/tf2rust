from .node import Node


class MultiplyNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)

    def apply_layer(self):
        args = " * ".join(
            ["&out_{}".format(layer_name) for layer_name in self.parents_name]
        )

        return [
            "let {declare_mut}out_{node.name}: Array{node.output_dimensions}<{node.dtype}> = {input};".format(
                declare_mut=self._format_mut(), node=self, input=args
            )
        ]
