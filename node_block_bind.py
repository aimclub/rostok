import node_render
import node

class NodeTerminal:
    def __init__(self, block_cls, *args, **kwargs):
        self.builder = None
        self.block_cls = block_cls
        self.args = args
        self.kwargs = kwargs

    def create_block(self):
        if self.builder is None:
            raise Exception('Set builder first')
        return self.block_cls(self.builder, *self.args, **self.kwargs)
