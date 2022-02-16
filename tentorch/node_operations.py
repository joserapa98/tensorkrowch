# split, svd, qr, rq, etc.
# contract, contract_between, batched_contract_between, einsum, etc.

# TODO: no se usa
def einsum(string: Text,
           *nodes: Sequence[Union[torch.Tensor, AbstractNode]]) -> AbstractNode:
    new_tensor = torch.einsum(string,
                              *tuple(map(lambda x: x.tensor if isinstance(x, AbstractNode) else x, nodes)))
    new_node = Node(tensor=new_tensor)

    out_string = string.split('->')[1]
    strings = string.split('->')[0].split(',')

    i = 0
    for j, string in enumerate(strings):
        if isinstance(nodes[j], AbstractNode):
            for k, char in enumerate(string):
                if i < len(out_string) and out_string[i] == char:
                    new_node.add_edge(nodes[j][k], i, override=True)
                    i += 1
    return new_node


def contract(edge: AbstractEdge) -> AbstractNode:
    nodes = [edge.node1, edge.node2]
    axes = [edge.axis1, edge.axis2]

    index = 1
    input_strings = []
    output_string = ''
    matrix_string = ''
    for i, node in enumerate(nodes):
        if (i == 1) and (nodes[1] == nodes[0]):
            break
        string = ''
        for j, _ in enumerate(node.shape):
            if j == axes[i].num:
                string += _VALID_SUBSCRIPTS[0]
                if isinstance(edge, ParamEdge):
                    matrix_string = 2 * _VALID_SUBSCRIPTS[0]
            else:
                string += _VALID_SUBSCRIPTS[index]
                output_string += _VALID_SUBSCRIPTS[index]
                index += 1
        input_strings.append(string)

    if len(input_strings) == 1:
        input_string = ''.join(input_strings[0])
        if isinstance(edge, ParamEdge):
            einsum_string = input_string + ',' + matrix_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, nodes[0].tensor, edge.matrix)
        else:
            einsum_string = input_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, nodes[0].tensor)
        # TODO: check names -> method in network for repeated names
        name = f'{nodes[0].name}![{axes[0].name}]'
    else:
        input_string_0 = ''.join(input_strings[0])
        input_string_1 = ''.join(input_strings[1])
        if isinstance(edge, ParamEdge):
            einsum_string = input_string_0 + ',' + matrix_string + ',' \
                            + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, nodes[0].tensor, edge.matrix, nodes[1].tensor)
        else:
            einsum_string = input_string_0 + ',' + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, nodes[0].tensor, nodes[1].tensor)
        name = f'{nodes[0].name},{nodes[1].name}![{axes[0].name},{axes[1].name}]'

    axes_names = []
    edges = []
    i = 0
    for j, string in enumerate(input_strings):
        for k, char in enumerate(string):
            if i < len(output_string):
                if output_string[i] == char:
                    axes_names.append(nodes[j].axes[k].name)
                    edges.append(nodes[j][k])
                    i += 1
            else:
                break
        if i >= len(output_string):
            break

    # TODO: eliminate previous nodes from network??
    new_node = Node(axes_names=axes_names,
                    name=name,
                    network=nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor)
    new_node.edges = edges
    return new_node


def get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
    edges = list()
    for edge in node1.edges:
        if (edge.node1 == node1) and (edge.node2 == node2):
            edges.append(edge)
        elif (edge.node1 == node2) and (edge.node2 == node1):
            edges.append(edge)
    return edges


def contract_between(node1: AbstractNode, node2: AbstractNode) -> AbstractNode:
    shared_edges = get_shared_edges(node1, node2)
    if not shared_edges:
        raise ValueError(f'No edges found between nodes {node1} and {node2}')

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges, _VALID_SUBSCRIPTS[:n_shared]))

    index = n_shared
    input_strings = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for node in [node1, node2]:
        if (node is node1) and (node1 is node2):
            break
        string = ''
        matrix_string = ''
        for edge in node.edges:
            if edge in shared_edges:
                string += shared_subscripts[edge]
                if isinstance(edge, ParamEdge):
                    matrices.append(edge.matrix)
                    matrix_string = 2 * shared_subscripts[edge]
            else:
                string += _VALID_SUBSCRIPTS[index]
                output_string += _VALID_SUBSCRIPTS[index]
                index += 1
        input_strings.append(string)
        matrices_strings.append(matrix_string)

    matrices_string = ','.join(matrices_strings)
    if len(input_strings) == 1:
        input_string = ''.join(input_strings[0])
        if len(matrices) > 0:
            einsum_string = input_string + ',' + matrices_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices)
        else:
            einsum_string = input_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor)
        name = f'{node1.name}'
    else:
        input_string_0 = ''.join(input_strings[0])
        input_string_1 = ''.join(input_strings[1])
        if len(matrices) > 0:
            einsum_string = input_string_0 + ',' + matrices_string + ',' \
                            + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices, node2.tensor)
        else:
            einsum_string = input_string_0 + ',' + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, node2.tensor)
        name = f'{node1.name}@{node2.name}'

    axes_names = []
    edges = []
    nodes = [node1, node2]
    i = 0
    for j, string in enumerate(input_strings):
        for k, char in enumerate(string):
            if i < len(output_string):
                if output_string[i] == char:
                    axes_names.append(nodes[j].axes[k].name)
                    edges.append(nodes[j][k])
                    i += 1
            else:
                break
        if i >= len(output_string):
            break

    new_node = Node(axes_names=axes_names,
                    name=name,
                    network=nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor)
    new_node.edges = edges
    return new_node


# TODO: deberíamos permitir contraer varios nodos a la vez, como un einsum,
#  para que vaya más optimizado. En un tree tendremos nodos iguales que apilaremos,
#  y cada nodo va conectado a otros 3, 4 nodos (los que sean). Así que hay que
#  contraer esas 3, 4 pilas de input con la pila de los tensores.
def batched_contract_between(node1: AbstractNode,
                             node2: AbstractNode,
                             batch_edge1: AbstractEdge,
                             batch_edge2: AbstractEdge) -> AbstractNode:
    """
    Contract between that supports one batch edge in each node.

    Uses einsum property: 'bij,bjk->bik'.

    Args:
        node1: First node to contract.
        node2: Second node to contract.
        batch_edge1: The edge of node1 that corresponds to its batch index.
        batch_edge2: The edge of node2 that corresponds to its batch index.

    Returns:
        new_node: Result of the contraction. This node has by default batch_edge1
        as its batch edge. Its edges are in order of the dangling edges of
        node1 followed by the dangling edges of node2.
    """
    # Stop computing if both nodes are the same
    if node1 is node2:
        raise ValueError(f'Cannot perform batched contraction between '
                         f'node {node1} and itself')

    shared_edges = get_shared_edges(node1, node2)
    if not shared_edges:
        raise ValueError(f'No edges found between nodes {node1} and {node2}')

    if batch_edge1 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge1} is shared between the nodes')
    if batch_edge2 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge2} is shared between the nodes')

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges, _VALID_SUBSCRIPTS[:n_shared]))

    index = n_shared + 1
    input_strings = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for node, batch_edge in zip([node1, node2], [batch_edge1, batch_edge2]):
        string = ''
        matrix_string = ''
        for edge in node.edges:
            if edge in shared_edges:
                string += shared_subscripts[edge]
                if isinstance(edge, ParamEdge):
                    matrices.append(edge.matrix)
                    matrix_string = 2 * shared_subscripts[edge]
            elif edge is batch_edge:
                string += _VALID_SUBSCRIPTS[n_shared]
                if node is node1:
                    output_string += _VALID_SUBSCRIPTS[n_shared]
            else:
                string += _VALID_SUBSCRIPTS[index]
                output_string += _VALID_SUBSCRIPTS[index]
                index += 1
        input_strings.append(string)
        matrices_strings.append(matrix_string)

    matrices_string = ','.join(matrices_strings)
    if len(input_strings) == 1:
        input_string = ''.join(input_strings[0])
        if len(matrices) > 0:
            einsum_string = input_string + ',' + matrices_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices)
        else:
            einsum_string = input_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor)
        name = f'{node1.name}'
    else:
        input_string_0 = ''.join(input_strings[0])
        input_string_1 = ''.join(input_strings[1])
        if len(matrices) > 0:
            einsum_string = input_string_0 + ',' + matrices_string + ',' \
                            + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices, node2.tensor)
        else:
            einsum_string = input_string_0 + ',' + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, node2.tensor)
        name = f'{node1.name}@{node2.name}'

    axes_names = []
    edges = []
    nodes = [node1, node2]
    i = 0
    for j, string in enumerate(input_strings):
        for k, char in enumerate(string):
            if i < len(output_string):
                if output_string[i] == char:
                    axes_names.append(nodes[j].axes[k].name)
                    edges.append(nodes[j][k])
                    i += 1
            else:
                break
        if i >= len(output_string):
            break

    new_node = Node(axes_names=axes_names,
                    name=name,
                    network=nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor)
    new_node.edges = edges
    return new_node


# stack, unbind, stacknode, stackedge, stacked_contraction


# TODO: implement this, ignore this at the moment
class StackNode(Node):
    def __init__(self,
                 nodes_list: List[AbstractNode],
                 dim: int,
                 shape: Optional[Union[int, Sequence[int], torch.Size]] = None,
                 axis_names: Optional[Sequence[Text]] = None,
                 network: Optional['TensorNetwork'] = None,
                 name: Optional[Text] = None,
                 tensor: Optional[torch.Tensor] = None,
                 param_edges: bool = True) -> None:

        tensors_list = list(map(lambda x: x.tensor, nodes_list))
        tensor = torch.stack(tensors_list, dim=dim)

        self.nodes_list = nodes_list
        self.tensor = tensor
        self.stacked_dim = dim

        self.edges_dict = dict()
        j = 0
        for node in nodes_list:
            for i, edge in enumerate(node.edges):
                if i >= self.stacked_dim:
                    j = 1
                if (i + j) in self.edges_dict:
                    self.edges_dict[(i + j)].append(edge)
                else:
                    self.edges_dict[(i + j)] = [edge]

        super().__init__(tensor=self.tensor)

    @staticmethod
    def set_tensor_format(tensor):
        return tensor

    def make_edge(self, axis: Axis):
        if axis == self.stacked_dim:
            return Edge(node1=self, axis1=axis)
        return StackEdge(self.edges_dict[axis], node1=self, axis1=axis)


# TODO: ignore this at the moment
class StackEdge(Edge):
    """
    Edge que es lista de varios Edges. Se usa para un StackNode

    No hacer cambios de tipos, simplemente que este Edge lleve
    parámetros adicionales de los Edges que está aglutinando
    y su vecinos y tal. Pero que no sea en sí mismo una lista
    de Edges, que eso lo lía todo.
    """

    def __init__(self,
                 edges_list: List['AbstractEdge'],
                 node1: 'AbstractNode',
                 axis1: int,
                 name: Optional[str] = None,
                 node2: Optional['AbstractNode'] = None,
                 axis2: Optional[int] = None,
                 shift: Optional[int] = None,
                 slope: Optional[int] = None) -> None:
        # TODO: edges in list must have same size and dimension
        self.edges_list = edges_list
        super().__init__(node1, axis1, name, node2, axis2, shift, slope)

    def create_parameters(self, shift, slope):
        return None, None, None

    def dim(self):
        """
        Si es ParamEdge se mide en función de sus parámetros la dimensión
        """
        return None

    def create_matrix(self, dim):
        """
        Eye for Edge, Parameter for ParamEdge
        """
        return None

    def __xor__(self, other: 'AbstractEdge') -> 'AbstractEdge':
        return None


def stack():
    pass


def unbind():
    pass
