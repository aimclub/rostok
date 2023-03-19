from dataclasses import dataclass


class BlockWrapper:
    """Class is interface between node and interpretation in simulation.

    The interface allows you to create an interpretation of terminal nodes in the simulation.
    Interpretation classes is in :py:mod:`node_render`.
    The instance must be specified when creating the node.
    When assembling a robot from a graph, an object is created by the 
    :py:meth:`BlockWrapper.create_block` method.
    When the object is created, the desired arguments of the interpretation object are set.

    Args:
        block_cls: Interpretation class of node in simulation
        args: Arguments py:attr:`BlockWrapper.block_cls`
        kwargs: Additional arguments py:attr:`BlockWrapper.block_cls`
    """

    def __init__(self, block_cls, *args, **kwargs):
        self.block_cls = block_cls
        self.args = args
        self.kwargs = kwargs

    def create_block(self):
        return self.block_cls(*self.args, **self.kwargs)


@dataclass
class Node:
    """Contains information about the label and :py:class:`BlockWrapper`,
    which is the physical representation of the node in the simulator
    """
    label: str = "*"
    is_terminal: bool = False

    # None for non-terminal nodes
    block_wrapper: BlockWrapper = None

    def __hash__(self) -> int:
        return hash(str(self.label) + str(self.is_terminal))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            raise Exception(
                "Wrong type of comparable object. Must be Node instead {wrong_type}".format(
                    wrong_type=type(__o)))
        return self.label == __o.label

@dataclass
class WrapperTuple:
    """ The return type is used to build the Robot.
        Id - from the generated graph
    """
    id: int
    block_wrapper: BlockWrapper  # Set default value



