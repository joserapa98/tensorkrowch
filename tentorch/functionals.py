from tentorch.network_components import Node


class Foo:
    def __init__(self):
        print('Creating operation foo')
        return

    def op(self, node1: Node, node2: Node):
        print('Operating node1 and node2')
        return
