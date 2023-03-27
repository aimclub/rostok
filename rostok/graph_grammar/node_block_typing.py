from rostok.graph_grammar.node import Node
from rostok.block_builder_chrono.block_classes import ChronoRevolveJoint, BlockBody, ChronoTransform


class NodeFeatures:

    @staticmethod
    def is_joint(node: Node):
        return node.block_blueprint.cls is ChronoRevolveJoint

    @staticmethod
    def is_body(node: Node):
        return node.block_blueprint.cls is BlockBody

    @staticmethod
    def is_transform(node: Node):
        return node.block_blueprint.cls is ChronoTransform