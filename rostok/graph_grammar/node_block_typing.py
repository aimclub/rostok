from rostok.graph_grammar.node import Node
from rostok.block_builder_api.block_blueprints import  BodyBlueprintType, JointBlueprintType, TransformBlueprintType


class NodeFeatures:
    @staticmethod
    def is_joint(node: Node):
        return issubclass(type(node.block_blueprint), JointBlueprintType)

    @staticmethod
    def is_body(node: Node):
        return issubclass(type(node.block_blueprint), BodyBlueprintType)

    @staticmethod
    def is_transform(node: Node):
        return issubclass(type(node.block_blueprint), TransformBlueprintType)