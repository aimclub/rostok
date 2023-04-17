from rostok.block_builder_api.block_blueprints import (BodyBlueprintType,
                                                       JointBlueprintType,
                                                       TransformBlueprintType)
from rostok.graph_grammar.node import GraphGrammar, Node


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

def get_joint_vector_from_graph(graph:GraphGrammar):
    joint_vector = []
    paths = graph.get_sorted_root_based_paths()
    for path in paths:
        for idx in path:
            if idx in joint_vector:
                continue
            else:
                if NodeFeatures.is_joint(graph.get_node_by_id(idx)):
                    joint_vector.append(idx)
    return joint_vector
