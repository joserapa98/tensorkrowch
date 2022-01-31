import torch
import torch.nn as nn

from typing import Union, Tuple, Optional, Text, List, overload
from abc import ABC, abstractmethod

Tensor = torch.Tensor


class AbstractNode(ABC):
    """
    Abstract node tiene todos los métodos que involucran
    operaciones con el tensor. La propiedad '.tensor' es un
    abstractmethod que tienen que definir obligatoriamente sus subclasses

    Definir todas las operaciones entre AbstractNodes, siempre se devuelve
    como resultado un Node. Los ParamNode nunca son resultado de una operación.

    Contraer dos nodos es in-place, es decir, cambiamos esos nodos
    por el resultado de la operación en la red. Poner un parametro
    in_place=True, que se pueda poner a False. Si es False, se devuelve
    una nueva red copiando todos los nodos menos donde se produce la
    operación, que se sustituye por el resultado. Esto pasa para Nodes,
    para ParamNodes nunca puede ser in_place.
    """

    def __init__(self,
                 network: Optional['TensorNetwork'] = None,
                 shape: Tuple[int, ...] = None,
                 tensor: Optional[Tensor] = None,
                 name: Optional[Text] = None,
                 axis_names: Optional[List[Text]] = None) -> None:
        super().__init__()

        if shape is None:
            if tensor is None:
                raise ValueError('At least one of "shape" or "tensor" must be provided')
            else:
                shape = tensor.shape
        else:
            if tensor is None:
                tensor = torch.empty(shape)
            else:
                if tensor.shape != shape:
                    raise ValueError('The tensor\'s shape must match the shape parameter')

        self._tensor = self.set_tensor_format(tensor)
        self._edges = [
            self.create_edge(axis=i)
            for i, _ in enumerate(shape)]

        if name is None:
            name = '__unnamed_node__'
        self._name = name
        # Si hacemos que los edges vayan referenciados al name de cada node
        # tenemos que hacerlo oculto y que no se pueda cambiar

        if axis_names is None:
            axis_names = list()
            for i, _ in enumerate(shape):
                axis_names.append(f'__unnamed_axis__{i}')
        elif len(axis_names) != len(shape):
            raise ValueError('Number of axis names must match the number of axis')
        self._axis_names = axis_names

        if network is not None:
            network.add_node(self)
        self._network = network

    @property
    def network(self):
        return self._network

    @staticmethod
    @abstractmethod
    def set_tensor_format(tensor):
        """
        Define la forma de establecer el tensor del nodo. Puede ser
        un tensor normal o un parámetro.
        """
        pass

    @abstractmethod
    def create_edge(self, axis=None, name=None):
        """
        Define el tipo de edges que se crean para un nodo,
        pueden ser ParamEdges (para ParamNodes) o Edges (Nodes).

        Damos un nombre por defecto a los Edges al crear el Node??
        (en el bucle donde se crean)
        """
        pass

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = self.set_tensor_format(tensor)

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def edges(self):
        """
        Usamos esto para no tener un setter de edges y no poder cambiarlo
        a gusto. Si eso hacemos función de cambiar un edge del nodo por
        otro dado como input
        """
        return self._edges

    def add_edge(self,
                 edge: 'AbstractEdge',
                 axis: Union[int, Text],
                 override: bool = False) -> None:
        axis_num = self.get_axis_number(axis)
        if axis_num < 0 or axis_num >= len(self.shape):
            raise VaueError('Error')
        if not self.edges[axis_num].is_dangling() and not override:
            raise ValueError('Error')
        self.edges[axis_num] = edge

    def size(self, dim=None):
        return self.tensor.size(dim)

    def dims(self, dim=None):
        if dim is None:
            dims = torch.tensor((map(lambda edge: edge.dim(), self.edges)))
        else:
            dims = self.edges[dim].dim()
        return dims

    @property
    def name(self):
        return self._name

    @property
    def axis_names(self):
        return self._axis_names

    @staticmethod
    def _copy_tensor(rank, dimension):
        shape = (dimension,) * rank
        copy_tensor = torch.zeros(shape)
        i = torch.arange(dimension)
        copy_tensor[(i,) * rank] = 1.
        return copy_tensor

    def initialize(self, init_method=None):
        """
        Cuando no inicializamos el nodo con un tensor, y a posteriori
        queremos inicializar toda la Network, usamos initialize, que
        implementa los métodos de inicialización de cada nodo de la
        TensorNetwork. Habitualmente se le llama cuando se llame a
        'tensor_network.initialize()'
        """
        # if init_method == 'copy':
        #    tensor = self._copy_tensor()
        # else:
        tensor = torch.randn(self.shape) * std
        self.tensor = tensor

    def get_axis_number(self, axis: Union[int, Text]) -> int:
        if isinstance(axis, int):
            return axis
        try:
            return self.axis_names.index(axis)
        except ValueError as err:
            raise ValueError(f'Axis name {axis} not found for node {self}') from err

    def get_edge(self, axis: Union[int, Text]) -> 'AbstractEdge':
        axis_number = self.get_axis_number(axis)
        return self.edges[axis_number]

    @overload
    def __getitem__(self, key: slice) -> List['AbstractEdge']:
        pass

    @overload
    def __getitem__(self, key: Union[int, Text]) -> 'AbstractEdge':
        pass

    def __getitem__(self, key: Union[slice, int,
                                     Text]) -> Union[List['AbstractEdge'], 'AbstractEdge']:
        if isinstance(key, slice):
            return self.edges[slice]
        return self.get_edge(key)

    # @abstractmethod
    def __matmul__(self, other: 'AbstractNode') -> 'AbstractNode':
        """
        Tal vez mejor lo implementamos en cada tipo d enodo que queramos crear
        """
        pass

    def __repr__(self) -> Text:
        return (f'\n{self.__class__.__name__}(\n'
                f'name : {self.name}\n'
                f'tensor :\n{self.tensor.data!r}\n'
                f'edges : \n{self.edges!r})')


class Node(AbstractNode):
    """
    Estos son solo nodos intermedios para las operaciones, no creamos
    redes (el grafo) con ellos

        -tensor
        -edges
        -network
        -copy
        -overloaded operations
        -to_param_node
        -to_copy_node (if possible)
    """

    @staticmethod
    def set_tensor_format(tensor):
        """
        Comprobar si tensor es torch.Tensor
        """
        return tensor

    def create_edge(self, axis=None, name=None):
        return Edge(node1=self, axis1=axis, name=name)


class CopyNode(Node):
    """
    Node pero es delta de Kroenecker y tiene la contracción optimizada
    """
    pass


class ParamNode(AbstractNode, nn.Module):
    """
        -grad -> devuelve grad del tensor

        -tensor (parameter)
        -edges
        -network
        -copy
        -copy without parameter (to Node)
        -overloaded operations
        -to_normal_node
        -to_copy_node (if possible)
    """

    def __init__(self,
                 network: Optional['TensorNetwork'] = None,
                 shape: Tuple[int, ...] = None,
                 tensor: Optional[Tensor] = None,
                 name: Optional[Text] = None,
                 axis_names: Optional[List[Text]] = None) -> None:
        nn.Module.__init__(self)
        AbstractNode.__init__(self, network, shape, tensor, name, axis_names)

    @staticmethod
    def set_tensor_format(tensor):
        """
        Comprobar si tensor es torch.Tensor
        """
        return nn.Parameter(tensor)

    def create_edge(self, axis=None, name=None):
        return ParamEdge(node1=self, axis1=axis, name=name)


class StackNode(AbstractNode):
    def __init__(self,
                 nodes_list: List['AbstractNode'],
                 dim: int,
                 network: Optional['TensorNetwork'] = None,
                 shape: Tuple[int, ...] = None,
                 tensor: Optional[Tensor] = None,
                 name: Optional[Text] = None,
                 axis_names: Optional[List[Text]] = None) -> None:

        tensors_list = list(map(lambda x: x.tensor, nodes_list))
        tensor = torch.stack(tensors_list, dim=dim)

        self.nodes_list = nodes_list
        self.tensor = tensor
        self.dim = dim

        self.edges_dict = dict()
        j = 0
        for node in nodes_list:
            for i, edge in enumerate(node.edges):
                if i >= self.dim:
                    j = 1
                if (i + j) in self.edges_dict:
                    self.edges_dict[(i + j)].append(edge)
                else:
                    self.edges_dict[(i + j)] = [edge]

        super().__init__(tensor=self.tensor)

    @staticmethod
    def set_tensor_format(tensor):
        return tensor

    def create_edge(self, axis=None, name=None):
        if axis == self.dim:
            return Edge(node1=self, axis1=axis, name=name)
        return StackEdge(self.edges_dict[axis], node1=self, axis1=axis, name=name)


class AbstractEdge(ABC):

    def __init__(self,
                 node1: 'AbstractNode',
                 axis1: int,
                 name: Optional[str] = None,
                 node2: Optional['AbstractNode'] = None,
                 axis2: Optional[int] = None,
                 shift: Optional[int] = None,
                 slope: Optional[int] = None) -> None:
        super().__init__()

        if (node2 is None) != (axis2 is None):
            raise ValueError('node2 and axis2 must either be both None or both not be None')

        if node2 is not None:
            if node1.shape[axis1] != node2.shape[axis2]:
                raise ValueError('Shapes of axis1 and axis2 must match')

        self._nodes = [node1, node2]
        self._axis = [axis1, axis2]
        self._is_dangling = node2 is None

        shift, slope, sigmoid = self.create_parameters(shift, slope)
        self.shift = shift
        self.slope = slope
        self.sigmoid = sigmoid

        size = node1.shape[axis1]
        self._size = size
        self._matrix = self.create_matrix(size)

        if name is None:
            random = torch.randn(1)
            name = f'__unnamed_edge__{axis1}__{str(random)[-5:]}'
        self.name = name

    @abstractmethod
    def create_parameters(self, shift, slope):
        pass

    def is_dangling(self) -> bool:
        return self._is_dangling

    @property
    def node1(self) -> 'AbstractNode':
        return self._nodes[0]

    @property
    def node2(self) -> 'AbstractNode':
        return self._nodes[1]

    @property
    def axis1(self) -> int:
        return self._axis[0]

    @property
    def axis2(self) -> int:
        return self._axis[1]

    @property
    def matrix(self):
        """
        Esta propiedad no estoy seguro, puede que valga con saber
        la dimensión del Edge
        """
        return self._matrix

    def size(self):
        return self._size

    @abstractmethod
    def dim(self):
        """
        Si es ParamEdge se mide en función de sus parámetros la dimensión
        """
        pass

    @abstractmethod
    def create_matrix(self, dim):
        """
        Eye for Edge, Parameter for ParamEdge
        """
        pass

    @abstractmethod
    def __xor__(self, other: 'AbstractEdge') -> 'AbstractEdge':
        pass

    def __repr__(self) -> Text:
        if self.node1 is not None and self.node2 is not None:
            return (f'\n{self.__class__.__name__}(\n'
                    f'{self.node1.name!r}[{self.axis1}] -> '
                    f'{self.node2.name!r}[{self.axis2}]) ({self.name})\n')
        return f'\n{self.__class__.__name__}(Dangling Edge)[{self.axis1}] ({self.name}) \n'


class Edge(AbstractEdge):
    """
        -batch (bool)
    """

    def create_parameters(self, shift, slope):
        return None, None, None

    def dim(self):
        return self.size()

    def create_matrix(self, size):
        return torch.eye(size)

    @overload
    def __xor__(self, other: 'Edge') -> 'Edge':
        pass

    @overload
    def __xor__(self, other: 'ParamEdge') -> 'ParamEdge':
        pass

    def __xor__(self, other: Union['Edge',
                                   'ParamEdge']) -> Union['Edge', 'ParamEdge']:
        global new_edge
        if not self.is_dangling() or not other.is_dangling():
            raise ValueError('Both edges must be dangling edges')
        if self is other:
            raise ValueError('Given edges cannot be the same')
        if self.size() != self.size():
            raise ValueError('Given edges must have the same size')
        if self.dim() != self.dim():
            raise ValueError('Given edges must have the same dimension')

        node1 = self.node1
        node2 = other.node1
        axis1_num = node1.get_axis_number(self.axis1)
        axis2_num = node2.get_axis_number(other.axis1)

        net1 = node1.network
        net2 = node2.network
        if net1 != net2:
            raise ValueError('Both nodes must be within the same Tensor Network')

        if isinstance(other, Edge):
            new_edge = Edge(node1=node1,
                            axis1=axis1_num,
                            node2=node2,
                            axis2=axis2_num)
        if isinstance(other, ParamEdge):
            shift = other.shift
            slope = other.slope
            new_edge = ParamEdge(node1=node1,
                                 axis1=axis1_num,
                                 node2=node2,
                                 axis2=axis2_num,
                                 shift=shift,
                                 slope=slope)

            if net1 is not None:
                net1._remove_param_edge(other)
                net1._add_param_edge(new_edge)

        node1.add_edge(new_edge, axis1_num, override=True)
        node2.add_edge(new_edge, axis2_num, override=True)
        return new_edge


class ParamEdge(AbstractEdge, nn.Module):
    """
        -batch (bool)
        -grad -> devuelve tupla de grad de shift y slope
    """

    def __init__(self,
                 node1: 'AbstractNode',
                 axis1: int,
                 name: Optional[str] = None,
                 node2: Optional['AbstractNode'] = None,
                 axis2: Optional[int] = None,
                 shift: Optional[int] = None,
                 slope: Optional[int] = None) -> None:
        nn.Module.__init__(self)

        if shift is None:
            shift = -1.
        if slope is None:
            slope = 10.
        # Empezamos lo más parecido a la identidad

        AbstractEdge.__init__(self, node1, axis1, name, node2, axis2, shift, slope)

    def create_parameters(self, shift, slope):
        shift = nn.Parameter(torch.tensor(float(shift)))
        slope = nn.Parameter(torch.tensor(float(slope)))
        sigmoid = nn.Sigmoid()
        return shift, slope, sigmoid

    def dim(self):
        size = self.matrix.shape[0]
        i = torch.arange(size)
        signs = torch.sign(
            self.sigmoid(self.slope * (i - self.shift)) - 0.5)
        dim = torch.where(signs == 1, signs,
                          torch.zeros(signs.shape)).sum()
        return int(dim)

    def create_matrix(self, size):
        matrix = torch.zeros((size, size))
        i = torch.arange(size)
        matrix[(i, i)] = self.sigmoid(self.slope * (i - self.shift))
        return matrix

    def __xor__(self, other: 'AbstractEdge') -> 'ParamEdge':
        if not self.is_dangling() or not other.is_dangling():
            raise ValueError('Both edges must be dangling edges')
        if self is other:
            raise ValueError('Given edges cannot be the same')
        if self.size() != self.size():
            raise ValueError('Given edges must have the same size')
        if self.dim() != self.dim():
            raise ValueError('Given edges must have the same dimension')

        node1 = self.node1
        node2 = other.node1
        axis1_num = node1.get_axis_number(self.axis1)
        axis2_num = node2.get_axis_number(other.axis1)

        net1 = node1.network
        net2 = node2.network
        # TODO:
        # Si alguna net es None, ponerle al otro la net del primero
        if net1 != net2:
            raise ValueError('Both nodes must be within the same Tensor Network')

        shift = self.shift
        slope = self.slope
        new_edge = ParamEdge(node1=node1,
                             axis1=axis1_num,
                             node2=node2,
                             axis2=axis2_num,
                             shift=shift,
                             slope=slope)

        if net1 is not None:
            net1._remove_param_edge(self)
            if isinstance(other, ParamEdge):
                net1._remove_param_edge(other)
            net1._add_param_edge(new_edge)

        node1.add_edge(new_edge, axis1_num, override=True)
        node2.add_edge(new_edge, axis2_num, override=True)
        return new_edge


class StackEdge(AbstractEdge):
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


class TensorNetwork(nn.Module):
    """
    Al contraer una red se crea una Network auxiliar formada por Nodes en lugar
    de ParamNodes y Edges en lugar de ParamEdges. Ahí se van guardando todos los
    nodos auxiliares, resultados de operaciones intermedias, se termina de contraer
    la red y se devuelve el resultado

    Formado opr AbstractNodes y AbstractEdges

        -nodes (add modules, del modules)
        -edges
        -forward (create nodes for input data and generate the final
            network to be contracted)
        -contraction of network (identify similar nodes to stack them
            and optimize)
        -to_tensornetwork (returns a dict or list with all the nodes
            with edges connected as the original network)
    """

    def __init__(self):
        super().__init__()

        self._nodes = []

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        edges_list = list()
        for node in self.nodes:
            for edge in node.edges:
                edges_list.append(edge)
        return edges_list

    def _add_param_edge(self, edge: 'ParamEdge') -> None:
        self.add_module(edge.name, edge)

    def _remove_param_edge(self, edge: 'ParamEdge') -> None:
        delattr(self, edge.name)

    def add_node(self, node: Node) -> None:
        """

        :rtype: object
        """
        # TODO:
        # if node.network != self:
        #    node.change_network(self)
        if isinstance(node, ParamNode):
            self.add_module(node.name, node)
            for edge in node.edges:
                assert isinstance(edge, ParamEdge)  ###...
                self.add_module(edge.name, edge)
        self._nodes.append(node)

    def add_nodes_from(self, nodes_list):
        for name, node in nodes_list:
            self.add_node(name, node)

    def connect_nodes(nodes_list, axis_list):
        if len(nodes_list) == 2 and len(axis_list) == 2:
            nodes_list[0][axis_list[0]] ^ nodes_list[1][axis_list[1]]
        else:
            raise ValueError('Both nodes_list and axis_list must have length 2')

    def initialize(self, *args):
        for child in self.children():
            if isinstance(child, AbstractNode):
                child.initialize(*args)
            # Los Edges se inicializan solos

    # def remove_node(self, name) -> None:
    #    delattr(self, name)
    #    self._nodes.remove(name)

    # def add_data(self, data):
    #    pass

    # @abstractmethod
    def contract_network(self):
        pass

    # def forward(self, data):
    #    aux_net = self.add_data(data)
    #    result = aux_net.contract_network()
    #    self.clear_op_network()
    #    return result


class MPS(TensorNetwork):
    """
        optimized operations
    """

    def __init__(self):
        super().__init__()

        # node = Node(net=self)
        # ...

    # En cada función restringir el tipo de Nodos que se pueden usar,
    # para ser siempre MPS


class Tree(TensorNetwork):
    """
        optimized operations
    """
    # Lo mismo que en MPS
    pass


_VALID_SUBSCRIPTS = list(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


def einsum(string: Text, nodes: List[Union[Tensor, 'AbstractNode']]) -> 'AbstractNode':
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


def contract(edge: 'AbstractEdge'):
    # TODO:
    # Podemos hacer aquí lo de añadir al string de einsum las matrices
    nodes = [edge.node1, edge.node2]
    axis = [edge.axis1, edge.axis2]
    assert node1 != node2

    next_subscript = 1
    input_strings = list()
    output_string = list()
    for i, node in enumerate(nodes):
        string = list()
        for j, _ in enumerate(node.shape):
            if j == axis[i]:
                string.append(_VALID_SUBSCRIPTS[0])
                if isinstance(edge, ParamEdge):
                    matrix_string = _VALID_SUBSCRIPTS[0] + _VALID_SUBSCRIPTS[0]
            else:
                string.append(_VALID_SUBSCRIPTS[next_subscript])
                output_string.append(_VALID_SUBSCRIPTS[next_subscript])
                next_subscript += 1
        input_strings.append(string)

    string1 = ''.join(input_strings[0])
    string2 = ''.join(input_strings[1])
    out_string = ''.join(output_strings)

    if isinstance(edge, ParamEdge):
        einsum_string = string1 + ',' + matrix_string + ',' + string2 + '->' + out_string
        new_node = einsum(einsum_string, nodes[0], edge.matrix, nodes[1])
    else:
        einsum_string = string1 + ',' + string2 + '->' + out_string
        new_node = einsum(einsum_string, nodes[0], nodes[1])

    return new_node


def get_shared_edges(node1, node2):
    list_edges = list()
    for edge in node1.edges:
        if edge.node1 == node1 and edge.node2 == node2:
            list_edges.append(edge)
        elif edge.node1 == node2 and edge.node2 == node1:
            list_edges.append(edge)

    return list_edges


def contract_between(node1: 'AbstractNode', node2: 'AbstractNode'):
    # TODO:
    # En lugar de hacer un bucle de contract, hacer un solo string y un solo
    # einsum para qeu esté más optimizado
    _VALID_SUBSCRIPTS = list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Stop computing if both nodes are the same
    if node1 is node2:
        raise ValueError(f'Cannot perform batched contraction between '
                         f'node {node1} and itself')

    # Computing shared edges
    shared_edges = get_shared_edges(node1, node2)
    # Stop computing if there are no shared edges
    if not shared_edges:  # shared_edges set is True if it has some element
        raise ValueError(f'No edges found between ndoes '
                         f'{node1} and {node2}')

    n_shared = len(shared_edges)
    # Create dictionary where each edge is a key, and each value is a letter
    shared_subscripts = dict(zip(shared_edges, _VALID_SUBSCRIPTS[:n_shared]))

    matrices_strings = []
    matrices = []
    res_string, string = [], []
    index = n_shared + 1
    # Loop in both (node1, batch_edge1) and (node2, batch_edge2)
    for node in [node1, node2]:
        # Append empty element for each loop, that is, string = [[],[]]
        string.append([])
        for edge in node.edges:
            # If edge is shared, add its letter to string
            if edge in shared_edges:
                string[-1].append(shared_subscripts[edge])
                if isinstance(edge, ParamEdge):
                    matrices_strings.append(shared_subscripts[edge] + shared_subscripts[edge])
                    matrices.append(edge.matrix)
            # If edge is neither shared or batch, add second next available new letter
            else:
                string[-1].append(_VALID_SUBSCRIPTS[index])
                res_string.append(_VALID_SUBSCRIPTS[index])
                index += 1

    string1 = ''.join(string[0])
    string2 = ''.join(string[1])
    mat_string = ','.join(matrices_strings)
    res_string = ''.join(res_string)

    # einsum_string = ''.join([string1, ',', string2, '->', res_string])
    if isinstance(edge, ParamEdge):
        einsum_string = string1 + ',' + mat_string + ',' + string2 + '->' + res_string
    else:
        einsum_string = string1 + ',' + string2 + '->' + res_string

    new_node = einsum(einsum_string, [node1] + matrices + [node2])

    return new_node


def batched_contract_between(node1: AbstractNode, node2: AbstractNode, batch_edge1: AbstractEdge,
                             batch_edge2: AbstractEdge) -> AbstractNode:
    """
    Contract between that supports one batch edge in each node.

    Uses einsum property: 'bij,bjk->bik'.

    Args:
        node1: First node to contract.
        node2: Second node to contract.
        batch_edge1: The edge of node1 that correpond to its batch index.
        batch_edge2: The edge of node2 that correpond to its batch index.

    Returns:
        new_node: Result of the contraction. This node has by default batch_edge1
        as its batch edge. Its edges are in order of the dangling edges of
        node1 followed by the dangling edges of node2.
    """

    _VALID_SUBSCRIPTS = list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Stop computing if both nodes are the same
    if node1 is node2:
        raise ValueError(f'Cannot perform batched contraction between '
                         f'node {node1} and itself')

    # Computing shared edges
    shared_edges = get_shared_edges(node1, node2)
    # Stop computing if there are no shared edges
    if not shared_edges:  # shared_edges set is True if it has some element
        raise ValueError(f'No edges found between ndoes '
                         f'{node1} and {node2}')

    # Stop computing if batch edges are in shared edges
    if batch_edge1 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge1} is shared between the nodes')
    if batch_edge2 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge2} is shared between the nodes')

    n_shared = len(shared_edges)
    # Create dictionary where each edge is a key, and each value is a letter
    shared_subscripts = dict(zip(shared_edges, _VALID_SUBSCRIPTS[:n_shared]))

    matrices_strings = []
    matrices = []
    res_string, string = [], []
    index = n_shared + 1
    # Loop in both (node1, batch_edge1) and (node2, batch_edge2)
    for node, batch_edge in zip([node1, node2], [batch_edge1, batch_edge2]):
        # Append empty element for each loop, that is, string = [[],[]]
        string.append([])
        for edge in node.edges:
            # If edge is shared, add its letter to string
            if edge in shared_edges:
                string[-1].append(shared_subscripts[edge])
                if isinstance(edge, ParamEdge):
                    matrices_strings.append(shared_subscripts[edge] + shared_subscripts[edge])
                    matrices.append(edge.matrix)
            # If edge is batch_edge, add next available letter
            elif edge is batch_edge:
                string[-1].append(_VALID_SUBSCRIPTS[n_shared])
                if node is node1:
                    res_string.append(_VALID_SUBSCRIPTS[n_shared])
            # If edge is neither shared or batch, add second next available new letter
            else:
                string[-1].append(_VALID_SUBSCRIPTS[index])
                res_string.append(_VALID_SUBSCRIPTS[index])
                index += 1

    string1 = ''.join(string[0])
    string2 = ''.join(string[1])
    mat_string = ','.join(matrices_strings)
    res_string = ''.join(res_string)

    if isinstance(edge, ParamEdge):
        einsum_string = string1 + ',' + mat_string + ',' + string2 + '->' + res_string
    else:
        einsum_string = string1 + ',' + string2 + '->' + res_string

    new_node = einsum(einsum_string, [node1] + matrices + [node2])

    return new_node


def stack():
    pass


def unbind():
    pass


# Las de debajo ya no sirven
def aux_batched_contraction(edge, batch_edge1, batch_edge2):
    # TODO:
    # Como contract pero añadimos el character del batch, como Senaida
    """
    given a list of nodes or param_nodes, stack their tensors,
    compute the contraction, copy the structure of the given
    nodes changing their tensors with the result of the computation

    A lo mejor este no hace falta y nos vale solo con batched_contract_between.
    Si lo metemos a lo mejor es bueno añadir indicador en cada edge de si es batch,
    para no tener que indicarlo siempre como parámetro
    """
    pass


def aux_batched_contract_between(edge1, edge2, batch_edge1, batch_ege2):
    # TODO:
    # Como contract between pero añadiendo un edge de batch
    """
    given two nodes computes the contraction of every connected
    indices, and adapts the computation if there are batch edges
    in both nodes to make batched_contract_between
    """
    pass


def split_node():
    # SVD decomposition
    # Tucker decomposition
    # CP decomposition
    pass