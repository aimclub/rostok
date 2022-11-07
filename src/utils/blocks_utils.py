from engine.node_render import ChronoBody, ChronoTransform, ChronoRevolveJoint
from engine.node import Node
from enum import Enum


class CollisionGroup(int, Enum):
    Default = 0
    Robot = 1
    Object = 2
    World  = 3


def make_collide(body_list: list[ChronoBody], group_id: CollisionGroup, self_colide=False):
    if type(group_id) is  not CollisionGroup:
        raise Exception("group_id must be CollisionGroup. Instead {wrong_type}".format(wrong_type=type(group_id)))
    
    for body in body_list:
        colision_model = body.body.GetCollisionModel()
        colision_model.SetFamily(group_id)
        if not self_colide:
            colision_model.SetFamilyMaskNoCollisionWithFamily(group_id)
        body.body.SetCollide(True)


class NodeFeatures:
    @staticmethod
    def is_joint(node: Node):
        return node.block_wrapper.block_cls is ChronoRevolveJoint
    def is_body(node: Node):
        return node.block_wrapper.block_cls is ChronoBody
    def is_transform(node: Node):
        return node.block_wrapper.block_cls is ChronoTransform    