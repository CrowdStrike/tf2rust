import numpy as np

from .node import Node


class BatchNormalizationNode(Node):
    def __init__(self, layer_info, layer_weights):
        super().__init__(layer_info, layer_weights)
        self.epsilon = str(layer_info["config"]["epsilon"])

        # gamma, beta, moving_mean, moving_variance
        assert (
            len(self.weights_list) == 4
        ), "self_weights_list has length {}, but this should be 4".format(
            len(self.weights_list)
        )

        self.type = "tensorflow_layers::BatchNormalization"

    @staticmethod
    def get_weights(layer):
        array_length = len(layer.get_weights()[0])

        res = []
        order_weights = ["gamma", "beta", "moving_mean", "moving_variance"]
        for weight_name in order_weights:
            weight = None
            for w in layer.weights:
                if weight_name in w.name.lower():
                    weight = w.numpy()
                    break
            if weight is None:
                if weight_name == "gamma":
                    weight = np.ones((array_length,))
                elif weight_name == "beta":
                    weight = np.zeros((array_length,))
                else:
                    raise Exception(
                        "Expected gamma or beta, found: {}".format(weight_name)
                    )

            res.append(weight)

        return res

    def can_be_done_inplace(self):
        return True

    def initialize_layer(self):
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
        args = ", ".join([args, self.epsilon])

        layer_declaration = "let {node.name} = {node.type}::new({input});".format(
            node=self, input=args
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

        if self.inplace_op:
            return [
                "self.{node.name}.apply_mut(&mut out_{input});".format(
                    node=self, input=self.parents_name[0]
                )
            ]
        else:
            return [
                "let {declare_mut}out_{node.name} = self.{node.name}.apply(&out_{input});".format(
                    declare_mut=self._format_mut(),
                    node=self,
                    input=self.parents_name[0],
                )
            ]
