"""
This script contains:

    Operation Class:
        * Operation
        
    Tensor-like operations:
        * permute
        * permute_           (in-place)
        * tprod
        * mul
        * div
        * add
        * sub
        * renormalize
        * conj
        
    Node-like operations:
        * split
        * split_             (in-place)
        * svd                           (edge operation)
        * svd_               (in-place) (edge operation)
        * svdr                          (edge operation)
        * svdr_              (in-place) (edge operation)
        * qr                            (edge operation)
        * qr_                (in-place) (edge operation)
        * rq                            (edge operation)
        * rq_                (in-place) (edge operation)
        * contract_edges
        * contract                      (edge operation)
        * contract_          (in-place) (edge operation)
        * get_shared_edges
        * contract_between
        * contract_between_  (in-place)
        * stack
        * unbind
        * einsum
        * stacked_einsum
"""

import types
from typing import Callable
from numbers import Number

from itertools import starmap
import opt_einsum

from tensorkrowch.components import *
from tensorkrowch.utils import (inverse_permutation, is_permutation,
                                list2slice, permute_list)


def copy_func(f):
    """Returns a function with the same code, defaults, closure and name."""
    fn = types.FunctionType(f.__code__, f.__globals__, f.__name__,
                            f.__defaults__, f.__closure__)

    # In case f was given attrs (note this dict is a shallow copy)
    fn.__dict__.update(f.__dict__)
    return fn


###############################################################################
#                               OPERATION CLASS                               #
###############################################################################


class Operation:  # MARK: Operation
    """
    Class for node operations.
    
    A node operation is made up of two functions, the one that is executed the
    first time the operation is called and the one that is executed in every
    other call (with the same arguments). Both functions are usually similar,
    though the former computes extra things regarding the creation of the
    ``resultant`` nodes and some auxiliary operations whose result will be the
    same in every call (e.g. when contracting two nodes, maybe a permutation of
    the tensors should be first performed; how this permutation is carried out
    is always the same, though the tensors themselves are different).

    Parameters
    ----------
    name : str
        Name of the operation. It cannot coincide with another operation's name.
        Operation names can be checked via ``net.operations``.
    check_first : callable
        Function that checks if the operation has been called at least one time.
    fn_first : callable
        Function that is called the first time the operation is performed.
    fn_next : callable
        Function that is called the next times the operation is performed.
    """

    def __init__(self,
                 name: Text,
                 check_first: Callable,
                 fn_first: Callable,
                 fn_next: Callable) -> None:
        assert isinstance(check_first, Callable)
        assert isinstance(fn_first, Callable)
        assert isinstance(fn_next, Callable)
        self.fn_first = fn_first
        self.fn_next = fn_next
        self.check_first = check_first

        # Operations could be overriden
        TensorNetwork.operations[name] = self

    def __call__(self, *args, **kwargs):
        successor = self.check_first(*args, **kwargs)

        if successor is None:
            return self.fn_first(*args, **kwargs)
        else:
            return self.fn_next(successor, *args, **kwargs)


###############################################################################
#                           TENSOR-LIKE OPERATIONS                            #
###############################################################################

#################################   PERMUTE    ################################
# MARK: permute
def _check_first_permute(node: AbstractNode,
                         axes: Sequence[Ax]) -> Optional[Successor]:
    args = (node, tuple(axes))
    successors = node._successors.get('permute')
    if not successors:
        return None
    return successors.get(args)


def _permute_first(node: AbstractNode, axes: Sequence[Ax]) -> Node:
    axes_nums = []
    for axis in axes:
        axes_nums.append(node.get_axis_num(axis))

    if not is_permutation(list(range(len(axes_nums))), axes_nums):
        raise ValueError('The provided list of axis is not a permutation of the'
                         ' axes of the node')
    else:
        new_node = Node._create_resultant(
            axes_names=permute_list(node.axes_names, axes_nums),
            name='permute',
            network=node._network,
            tensor=node.tensor.permute(axes_nums),
            edges=permute_list(node._edges, axes_nums),
            node1_list=permute_list(node.is_node1(), axes_nums))

    # Create successor
    net = node._network
    args = (node, tuple(axes))
    successor = Successor(node_ref=node.node_ref(),
                          index=node._tensor_info['index'],
                          child=new_node,
                          hints=axes_nums)

    # Add successor to parent
    if 'permute' in node._successors:
        node._successors['permute'].update({args: successor})
    else:
        node._successors['permute'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('permute', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node._record_in_inverse_memory()

    return new_node


def _permute_next(successor: Successor,
                  node: AbstractNode,
                  axes: Sequence[Ax]) -> Node:
    # All arguments are mandatory though some might not be used
    new_tensor = node._direct_get_tensor(successor.node_ref,
                                         successor.index)
    new_tensor = new_tensor.permute(successor.hints)
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node._network._traced:
        node._check_inverse_memory(successor.node_ref)

    return child


permute_op = Operation('permute',
                       _check_first_permute,
                       _permute_first,
                       _permute_next)


def permute(node: AbstractNode, axes: Sequence[Ax]) -> Node:
    """
    Permutes the nodes' tensor, as well as its axes and edges to match the new
    shape.

    See `permute <https://pytorch.org/docs/stable/generated/torch.permute.html>`_
    in the **PyTorch** documentation.
    
    Nodes ``resultant`` from this operation are called ``"permute"``. The node
    that keeps information about the :class:`Successor` is ``node``.
    
    This operation is the same as :meth:`~AbstractNode.permute`.

    Parameters
    ----------
    node: Node or ParamNode
        Node whose tensor is to be permuted.
    axes: list[int, str or Axis]
        List of axes in the permuted order.
        
    Returns
    -------
    Node
    
    Examples
    --------
    >>> node = tk.randn((2, 5, 7))
    >>> result = tk.permute(node, (2, 0, 1))
    >>> result.shape
    torch.Size([7, 2, 5])
    """
    return permute_op(node, axes)


permute_node = copy_func(permute)
permute_node.__doc__ = \
    """
    Permutes the nodes' tensor, as well as its axes and edges to match the new
    shape.
    
    See `permute <https://pytorch.org/docs/stable/generated/torch.permute.html>`_
    in the **PyTorch** documentation.
    
    Nodes ``resultant`` from this operation are called ``"permute"``. The node
    that keeps information about the :class:`Successor` is ``self``.
    
    Parameters
    ----------
    axes: list[int, str or Axis]
        List of axes in the permuted order.
        
    Returns
    -------
    Node
        
    Examples
    --------
    >>> node = tk.randn((2, 5, 7))
    >>> result = node.permute((2, 0, 1))
    >>> result.shape
    torch.Size([7, 2, 5])
    """

AbstractNode.permute = permute_node


def permute_(node: AbstractNode, axes: Sequence[Ax]) -> Node:
    """
    Permutes the nodes' tensor, as well as its axes and edges to match the new
    shape (in-place).

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.

    See `permute <https://pytorch.org/docs/stable/generated/torch.permute.html>`_.
    
    Nodes ``resultant`` from this operation use the same name as ``node``.
    
    This operation is the same as :meth:`~AbstractNode.permute_`.

    Parameters
    ----------
    node: Node or ParamNode
        Node whose tensor is to be permuted.
    axes: list[int, str or Axis]
        List of axes in the permuted order.
        
    Returns
    -------
    Node
    
    Examples
    --------
    >>> node = tk.randn((2, 5, 7))
    >>> node = tk.permute_(node, (2, 0, 1))
    >>> node.shape
    torch.Size([7, 2, 5])
    """
    axes_nums = []
    for axis in axes:
        axes_nums.append(node.get_axis_num(axis))

    if not is_permutation(list(range(len(axes_nums))), axes_nums):
        raise ValueError('The provided list of axis is not a permutation of the'
                         ' axes of the node')
    else:
        new_node = Node(axes_names=permute_list(node.axes_names, axes_nums),
                        name=node._name,
                        override_node=True,
                        network=node._network,
                        override_edges=True,
                        tensor=node.tensor.permute(axes_nums).detach(),
                        edges=permute_list(node._edges, axes_nums),
                        node1_list=permute_list(node.is_node1(), axes_nums))

    return new_node


permute_node_ = copy_func(permute_)
permute_node_.__doc__ = \
    """
    Permutes the nodes' tensor, as well as its axes and edges to match the new
    shape (in-place).

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.

    See `permute <https://pytorch.org/docs/stable/generated/torch.permute.html>`_.
    
    Nodes ``resultant`` from this operation use the same name as ``node``.

    Parameters
    ----------
    axes: list[int, str or Axis]
        List of axes in the permuted order.
        
    Returns
    -------
    Node
        
    >>> node = tk.randn((2, 5, 7))
    >>> node = node.permute_((2, 0, 1))
    >>> node.shape
    torch.Size([7, 2, 5])
    """

AbstractNode.permute_ = permute_node_


##################################   TPROD    #################################
# MARK: tprod
def _check_first_tprod(node1: AbstractNode,
                       node2: AbstractNode) -> Optional[Successor]:
    args = (node1, node2)
    successors = node1._successors.get('tprod')
    if not successors:
        return None
    return successors.get(args)


def _tprod_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')
    if node2 in node1.neighbours():
        raise ValueError('Tensor product cannot be performed between connected'
                         ' nodes')

    new_tensor = torch.outer(node1.tensor.flatten(),
                             node2.tensor.flatten()).view(*(list(node1._shape) +
                                                            list(node2._shape)))
    new_node = Node._create_resultant(
        axes_names=node1.axes_names + node2.axes_names,
        name='tprod',
        network=node1._network,
        tensor=new_tensor,
        edges=node1._edges + node2._edges,
        node1_list=node1.is_node1() + node2.is_node1())

    # Create successor
    net = node1._network
    args = (node1, node2)
    successor = Successor(node_ref=(node1.node_ref(),
                                    node2.node_ref()),
                          index=(node1._tensor_info['index'],
                                 node2._tensor_info['index']),
                          child=new_node)

    # Add successor to parent
    if 'tprod' in node1._successors:
        node1._successors['tprod'].update({args: successor})
    else:
        node1._successors['tprod'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('tprod', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node1._record_in_inverse_memory()
        node2._record_in_inverse_memory()

    return new_node


def _tprod_next(successor: Successor,
                node1: AbstractNode,
                node2: AbstractNode) -> Node:
    tensor1 = node1._direct_get_tensor(successor.node_ref[0],
                                       successor.index[0])
    tensor2 = node2._direct_get_tensor(successor.node_ref[1],
                                       successor.index[1])
    new_tensor = torch.outer(tensor1.flatten(),
                             tensor2.flatten()).view(*(list(node1._shape) + 
                                                       list(node2._shape)))
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node1._network._traced:
        node1._check_inverse_memory(successor.node_ref)
        node2._check_inverse_memory(successor.node_ref)

    return child


tprod_op = Operation('tprod', _check_first_tprod, _tprod_first, _tprod_next)


def tprod(node1: AbstractNode, node2: AbstractNode) -> Node:
    """
    Tensor product between two nodes. It can also be performed using the
    operator ``%``.
    
    Nodes ``resultant`` from this operation are called ``"tprod"``. The node
    that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    node1 : Node or ParamNode
        First node to be multiplied. Its edges will appear first in the
        resultant node.
    node2 : Node or ParamNode
        Second node to be multiplied. Its edges will appear second in the
        resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((4, 5), network=net)
    >>> result = nodeA % nodeB
    >>> result.shape
    torch.Size([2, 3, 4, 5])
    """
    return tprod_op(node1, node2)


tprod_node = copy_func(tprod)
tprod_node.__doc__ = \
    """
    Tensor product between two nodes. It can also be performed using the
    operator ``%``.
    
    Nodes ``resultant`` from this operation are called ``"tprod"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : Node or ParamNode
        Second node to be multiplied. Its edges will appear second in the
        resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((4, 5), network=net)
    >>> result = nodeA.tprod(nodeB)
    >>> result.shape
    torch.Size([2, 3, 4, 5])
    """

AbstractNode.__mod__ = tprod_node


###################################   MUL    ##################################
# MARK: mul
def _check_first_mul(node1: AbstractNode,
                     node2: Union[AbstractNode,
                                  torch.Tensor,
                                  Number]) -> Optional[Successor]:
    if isinstance(node2, AbstractNode):
        args = (node1, node2)
    else:
        args = (node1,)
    successors = node1._successors.get('mul')
    if not successors:
        return None
    return successors.get(args)


def _mul_first(node1: AbstractNode,
               node2: Union[AbstractNode,
                            torch.Tensor,
                            Number]) -> Node:
    is_node2 = False
    if isinstance(node2, AbstractNode):
        is_node2 = True
        if node1._network != node2._network:
            raise ValueError('Nodes must be in the same network')

    if is_node2:
        new_tensor = node1.tensor * node2.tensor
    else:
        new_tensor = node1.tensor * node2
    
    new_node = Node._create_resultant(axes_names=node1.axes_names,
                                      name='mul',
                                      network=node1._network,
                                      tensor=new_tensor,
                                      edges=node1._edges,
                                      node1_list=node1.is_node1())

    # Create successor
    net = node1._network
    
    if is_node2:
        args = (node1, node2)
        successor = Successor(node_ref=(node1.node_ref(),
                                        node2.node_ref()),
                              index=(node1._tensor_info['index'],
                                     node2._tensor_info['index']),
                              child=new_node,
                              hints=is_node2)
    else:
        args = (node1,)
        successor = Successor(node_ref=(node1.node_ref(),),
                              index=(node1._tensor_info['index'],),
                              child=new_node,
                              hints=is_node2)

    # Add successor to parent
    if 'mul' in node1._successors:
        node1._successors['mul'].update({args: successor})
    else:
        node1._successors['mul'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('mul', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node1._record_in_inverse_memory()
        
        if is_node2:
            node2._record_in_inverse_memory()

    return new_node


def _mul_next(successor: Successor,
              node1: AbstractNode,
              node2: Union[AbstractNode,
                           torch.Tensor,
                           Number]) -> Node:
    is_node2 = successor.hints
    tensor1 = node1._direct_get_tensor(successor.node_ref[0],
                                       successor.index[0])
    if is_node2:
        tensor2 = node2._direct_get_tensor(successor.node_ref[1],
                                           successor.index[1])
    else:
        tensor2 = node2
    
    new_tensor = tensor1 * tensor2
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node1._network._traced:
        node1._check_inverse_memory(successor.node_ref[0])
        
        if is_node2:
            node2._check_inverse_memory(successor.node_ref[1])
    
    return child


mul_op = Operation('mul', _check_first_mul, _mul_first, _mul_next)


def mul(node1: AbstractNode,
        node2: Union[AbstractNode,
                     torch.Tensor,
                     Number]) -> Node:
    """
    Element-wise product between two nodes. It can also be performed using the
    operator ``*``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    multiplied by the ``node1`` tensor as ``node1.tensor * node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"mul"``. The node
    that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    node1 : Node or ParamNode
        First node to be multiplied. Its edges will appear in the resultant node.
    node2 : Node, ParamNode, torch.Tensor or number
        Second node to be multiplied. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA * nodeB
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA * tensorB
    >>> result.shape
    torch.Size([2, 3])
    """
    return mul_op(node1, node2)


mul_node = copy_func(mul)
mul_node.__doc__ = \
    """
    Element-wise product between two nodes. It can also be performed using the
    operator ``*``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    multiplied by the ``self`` tensor as ``self.tensor * node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"mul"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : Node, ParamNode, torch.Tensor or number
        Second node to be multiplied. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA.mul(nodeB)
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA.mul(tensorB)
    >>> result.shape
    torch.Size([2, 3])
    """

AbstractNode.__mul__ = mul_node


###################################   DIV    ##################################
# MARK: div
def _check_first_div(node1: AbstractNode,
                     node2: Union[AbstractNode,
                                  torch.Tensor,
                                  Number]) -> Optional[Successor]:
    if isinstance(node2, AbstractNode):
        args = (node1, node2)
    else:
        args = (node1,)
    successors = node1._successors.get('div')
    if not successors:
        return None
    return successors.get(args)


def _div_first(node1: AbstractNode,
               node2: Union[AbstractNode,
                            torch.Tensor,
                            Number]) -> Node:
    is_node2 = False
    if isinstance(node2, AbstractNode):
        is_node2 = True
        if node1._network != node2._network:
            raise ValueError('Nodes must be in the same network')

    if is_node2:
        new_tensor = node1.tensor / node2.tensor
    else:
        new_tensor = node1.tensor / node2
    
    new_node = Node._create_resultant(axes_names=node1.axes_names,
                                      name='div',
                                      network=node1._network,
                                      tensor=new_tensor,
                                      edges=node1._edges,
                                      node1_list=node1.is_node1())

    # Create successor
    net = node1._network
    
    if is_node2:
        args = (node1, node2)
        successor = Successor(node_ref=(node1.node_ref(),
                                        node2.node_ref()),
                              index=(node1._tensor_info['index'],
                                     node2._tensor_info['index']),
                              child=new_node,
                              hints=is_node2)
    else:
        args = (node1,)
        successor = Successor(node_ref=(node1.node_ref(),),
                              index=(node1._tensor_info['index'],),
                              child=new_node,
                              hints=is_node2)

    # Add successor to parent
    if 'div' in node1._successors:
        node1._successors['div'].update({args: successor})
    else:
        node1._successors['div'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('div', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node1._record_in_inverse_memory()
        
        if is_node2:
            node2._record_in_inverse_memory()

    return new_node


def _div_next(successor: Successor,
              node1: AbstractNode,
              node2: Union[AbstractNode,
                           torch.Tensor,
                           Number]) -> Node:
    is_node2 = successor.hints
    tensor1 = node1._direct_get_tensor(successor.node_ref[0],
                                       successor.index[0])
    if is_node2:
        tensor2 = node2._direct_get_tensor(successor.node_ref[1],
                                           successor.index[1])
    else:
        tensor2 = node2
    
    new_tensor = tensor1 / tensor2
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node1._network._traced:
        node1._check_inverse_memory(successor.node_ref[0])
        
        if is_node2:
            node2._check_inverse_memory(successor.node_ref[1])
    
    return child


div_op = Operation('div', _check_first_div, _div_first, _div_next)


def div(node1: AbstractNode,
        node2: Union[AbstractNode,
                     torch.Tensor,
                     Number]) -> Node:
    """
    Element-wise division between two nodes. It can also be performed using the
    operator ``/``.
    
    It also admits to take as ``node2`` a number or tensor, that will
    divide the ``node1`` tensor as ``node1.tensor / node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"div"``. The node
    that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    node1 : Node or ParamNode
        First node to be divided. Its edges will appear in the resultant node.
    node2 : Node, ParamNode, torch.Tensor or number
        Second node, the divisor. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA / nodeB
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA / tensorB
    >>> result.shape
    torch.Size([2, 3])
    """
    return div_op(node1, node2)


div_node = copy_func(div)
div_node.__doc__ = \
    """
    Element-wise division between two nodes. It can also be performed using the
    operator ``/``.
    
    It also admits to take as ``node2`` a number or tensor, that will
    divide the ``self`` tensor as ``self.tensor / node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"div"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : Node, ParamNode, torch.Tensor or number
        Second node, the divisor. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA.div(nodeB)
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA.div(tensorB)
    >>> result.shape
    torch.Size([2, 3])
    """

AbstractNode.__truediv__ = div_node


###################################   ADD    ##################################
# MARK: add
def _check_first_add(node1: AbstractNode,
                     node2: Union[AbstractNode,
                                  torch.Tensor,
                                  Number]) -> Optional[Successor]:
    if isinstance(node2, AbstractNode):
        args = (node1, node2)
    else:
        args = (node1,)
    successors = node1._successors.get('add')
    if not successors:
        return None
    return successors.get(args)


def _add_first(node1: AbstractNode,
               node2: Union[AbstractNode,
                            torch.Tensor,
                            Number]) -> Node:
    is_node2 = False
    if isinstance(node2, AbstractNode):
        is_node2 = True
        if node1._network != node2._network:
            raise ValueError('Nodes must be in the same network')

    if is_node2:
        new_tensor = node1.tensor + node2.tensor
    else:
        new_tensor = node1.tensor + node2
    
    new_node = Node._create_resultant(axes_names=node1.axes_names,
                                      name='add',
                                      network=node1._network,
                                      tensor=new_tensor,
                                      edges=node1._edges,
                                      node1_list=node1.is_node1())

    # Create successor
    net = node1._network
    
    if is_node2:
        args = (node1, node2)
        successor = Successor(node_ref=(node1.node_ref(),
                                        node2.node_ref()),
                              index=(node1._tensor_info['index'],
                                     node2._tensor_info['index']),
                              child=new_node,
                              hints=is_node2)
    else:
        args = (node1,)
        successor = Successor(node_ref=(node1.node_ref(),),
                              index=(node1._tensor_info['index'],),
                              child=new_node,
                              hints=is_node2)

    # Add successor to parent
    if 'add' in node1._successors:
        node1._successors['add'].update({args: successor})
    else:
        node1._successors['add'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('add', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node1._record_in_inverse_memory()
        
        if is_node2:
            node2._record_in_inverse_memory()

    return new_node


def _add_next(successor: Successor,
              node1: AbstractNode,
              node2: Union[AbstractNode,
                           torch.Tensor,
                           Number]) -> Node:
    is_node2 = successor.hints
    tensor1 = node1._direct_get_tensor(successor.node_ref[0],
                                       successor.index[0])
    if is_node2:
        tensor2 = node2._direct_get_tensor(successor.node_ref[1],
                                           successor.index[1])
    else:
        tensor2 = node2
    
    new_tensor = tensor1 + tensor2
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node1._network._traced:
        node1._check_inverse_memory(successor.node_ref[0])
        
        if is_node2:
            node2._check_inverse_memory(successor.node_ref[1])

    return child


add_op = Operation('add', _check_first_add, _add_first, _add_next)


def add(node1: AbstractNode,
        node2: Union[AbstractNode,
                     torch.Tensor,
                     Number]) -> Node:
    """
    Element-wise addition between two nodes. It can also be performed using the
    operator ``+``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    added to the ``node1`` tensor as ``node1.tensor + node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"add"``. The node
    that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    node1 : Node or ParamNode
        First node to be added. Its edges will appear in the resultant node.
    node2 : Node, ParamNode, torch.Tensor or numeric
        Second node to be added. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA + nodeB
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA + tensorB
    >>> result.shape
    torch.Size([2, 3])
    """
    return add_op(node1, node2)


add_node = copy_func(add)
add_node.__doc__ = \
    """
    Element-wise addition between two nodes. It can also be performed using the
    operator ``+``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    added to the ``self`` tensor as ``self.tensor + node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"add"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : Node, ParamNode, torch.Tensor or number
        Second node to be added. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA.add(nodeB)
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA.add(tensorB)
    >>> result.shape
    torch.Size([2, 3])
    """

AbstractNode.__add__ = add_node


###################################   SUB    ##################################
# MARK: sub
def _check_first_sub(node1: AbstractNode,
                     node2: Union[AbstractNode,
                                  torch.Tensor,
                                  Number]) -> Optional[Successor]:
    if isinstance(node2, AbstractNode):
        args = (node1, node2)
    else:
        args = (node1,)
    successors = node1._successors.get('sub')
    if not successors:
        return None
    return successors.get(args)


def _sub_first(node1: AbstractNode,
               node2: Union[AbstractNode,
                            torch.Tensor,
                            Number]) -> Node:
    is_node2 = False
    if isinstance(node2, AbstractNode):
        is_node2 = True
        if node1._network != node2._network:
            raise ValueError('Nodes must be in the same network')

    if is_node2:
        new_tensor = node1.tensor - node2.tensor
    else:
        new_tensor = node1.tensor - node2
    
    new_node = Node._create_resultant(axes_names=node1.axes_names,
                                      name='sub',
                                      network=node1._network,
                                      tensor=new_tensor,
                                      edges=node1._edges,
                                      node1_list=node1.is_node1())

    # Create successor
    net = node1._network
    
    if is_node2:
        args = (node1, node2)
        successor = Successor(node_ref=(node1.node_ref(),
                                        node2.node_ref()),
                              index=(node1._tensor_info['index'],
                                     node2._tensor_info['index']),
                              child=new_node,
                              hints=is_node2)
    else:
        args = (node1,)
        successor = Successor(node_ref=(node1.node_ref(),),
                              index=(node1._tensor_info['index'],),
                              child=new_node,
                              hints=is_node2)

    # Add successor to parent
    if 'sub' in node1._successors:
        node1._successors['sub'].update({args: successor})
    else:
        node1._successors['sub'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('sub', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node1._record_in_inverse_memory()
        
        if is_node2:
            node2._record_in_inverse_memory()

    return new_node


def _sub_next(successor: Successor,
              node1: AbstractNode,
              node2: Union[AbstractNode,
                           torch.Tensor,
                           Number]) -> Node:
    is_node2 = successor.hints
    tensor1 = node1._direct_get_tensor(successor.node_ref[0],
                                       successor.index[0])
    if is_node2:
        tensor2 = node2._direct_get_tensor(successor.node_ref[1],
                                           successor.index[1])
    else:
        tensor2 = node2
    
    new_tensor = tensor1 - tensor2
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node1._network._traced:
        node1._check_inverse_memory(successor.node_ref[0])
        
        if is_node2:
            node2._check_inverse_memory(successor.node_ref[1])

    return child


sub_op = Operation('sub', _check_first_sub, _sub_first, _sub_next)


def sub(node1: AbstractNode,
        node2: Union[AbstractNode,
                     torch.Tensor,
                     Number]) -> Node:
    """
    Element-wise subtraction between two nodes. It can also be performed using
    the operator ``-``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    subtracted from the ``node1`` tensor as ``node1.tensor - node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"sub"``. The node
    that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    node1 : Node or ParamNode
        First node, minuend . Its edges will appear in the resultant node.
    node2 : Node, ParamNode, torch.Tensor or number
        Second node, subtrahend. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA - nodeB
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA - tensorB
    >>> result.shape
    torch.Size([2, 3])
    """
    return sub_op(node1, node2)


sub_node = copy_func(sub)
sub_node.__doc__ = \
    """
    Element-wise subtraction between two nodes. It can also be performed using
    the operator ``-``.
    
    It also admits to take as ``node2`` a number or tensor, that will be
    subtracted from the ``self`` tensor as ``self.tensor - node2``. If this
    is used like this in the :meth:`~tensorkrowch.TensorNetwork.contract` method
    of a  :class:`~tensorkrowch.TensorNetwork`, this will have to be called
    explicitly to contract the network, rather than relying on its internal
    call via the :meth:`~tensorkrowch.TensorNetwork.forward`.
    
    Nodes ``resultant`` from this operation are called ``"sub"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : Node, ParamNode, torch.Tensor or number
        Second node, subtrahend. It can also be a number or tensor with
        appropiate shape.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> nodeB = tk.randn((2, 3), network=net)
    >>> result = nodeA.sub(nodeB)
    >>> result.shape
    torch.Size([2, 3])
    
    >>> net = tk.TensorNetwork()
    >>> nodeA = tk.randn((2, 3), network=net)
    >>> tensorB = torch.randn(2, 3)
    >>> result = nodeA.sub(tensorB)
    >>> result.shape
    torch.Size([2, 3])
    """

AbstractNode.__sub__ = sub_node


###############################   renormalize    ##############################
# MARK: renormalize
def _check_first_renormalize(
    node: AbstractNode,
    p: Union[int, float] = 2,
    axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Optional[Successor]:
    
    if isinstance(axis, (tuple, list)):
        axis = tuple(axis)
    args = (node, p, axis)
    successors = node._successors.get('renormalize')
    if not successors:
        return None
    return successors.get(args)


def _renormalize_first(
    node: AbstractNode,
    p: Union[int, float] = 2,
    axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Node:
    
    axis_num = []
    if axis is not None:
        if isinstance(axis, (tuple, list)):
            for ax in axis:
                axis_num.append(node.get_axis_num(ax))
            axis = tuple(axis)
        else:
            axis_num.append(node.get_axis_num(axis))
    
    norm = node.tensor.norm(p=p, dim=axis_num, keepdim=True)
    norm = torch.where(norm == 0., 1., norm)
    new_tensor = node.tensor / norm
    
    if isinstance(node, (StackNode, ParamStackNode)):
        new_node = StackNode._create_resultant(axes_names=node.axes_names,
                                               name='renormalize',
                                               network=node._network,
                                               tensor=new_tensor,
                                               edges=node._edges,
                                               node1_list=node.is_node1())
    else:
        new_node = Node._create_resultant(axes_names=node.axes_names,
                                          name='renormalize',
                                          network=node._network,
                                          tensor=new_tensor,
                                          edges=node._edges,
                                          node1_list=node.is_node1())
    
    # Create successor
    net = node._network
    args = (node, p, axis)
    successor = Successor(node_ref=node.node_ref(),
                          index=node._tensor_info['index'],
                          child=new_node,
                          hints=axis_num)

    # Add successor to parent
    if 'renormalize' in node._successors:
        node._successors['renormalize'].update({args: successor})
    else:
        node._successors['renormalize'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('renormalize', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node._record_in_inverse_memory()

    return new_node


def _renormalize_next(
    successor: Successor,
    node: AbstractNode,
    p: Union[int, float] = 2,
    axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Node:
    
    axis_num = successor.hints
    tensor = node._direct_get_tensor(successor.node_ref,
                                     successor.index)
    norm = tensor.norm(p=p, dim=axis_num, keepdim=True)
    norm = torch.where(norm == 0., 1., norm)
    new_tensor = tensor / norm
    
    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node._network._traced:
        node._check_inverse_memory(successor.node_ref)
    
    return child


renormalize_op = Operation('renormalize',
                           _check_first_renormalize,
                           _renormalize_first,
                           _renormalize_next)


def renormalize(
    node: AbstractNode,
    p: Union[int, float] = 2,
    axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Node:
    """
    Normalizes the node with the specified norm. That is, the tensor of ``node``
    is divided by its norm.
    
    Different norms can be taken, specifying the argument ``p``, and accross
    different dimensions, or node axes, specifying the argument ``axis``.
    
    See also `torch.norm() <https://pytorch.org/docs/stable/generated/torch.norm.html>`_.

    Parameters
    ----------
    node : Node or ParamNode
        Node that is to be renormalized.
    p : int, float
        The order of the norm.
    axis : int, str, Axis or list[int, str or Axis], optional
        Axis or sequence of axes over which to reduce.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn((3, 3))
    >>> renormA = tk.renormalize(nodeA)
    >>> renormA.norm()
    tensor(1.)
    """
    return renormalize_op(node, p, axis)


renormalize_node = copy_func(renormalize)
renormalize_node.__doc__ = \
    """
    Normalizes the node with the specified norm. That is, the tensor of ``node``
    is divided by its norm.
    
    Different norms can be taken, specifying the argument ``p``, and accross
    different dimensions, or node axes, specifying the argument ``axis``.
    
    See also `torch.norm() <https://pytorch.org/docs/stable/generated/torch.norm.html>`_.

    Parameters
    ----------
    p : int, float
        The order of the norm.
    axis : int, str, Axis or list[int, str or Axis], optional
        Axis or sequence of axes over which to reduce.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn((3, 3))
    >>> renormA = nodeA.renormalize()
    >>> renormA.norm()
    tensor(1.)
    """

AbstractNode.renormalize = renormalize_node


##################################   conj    ##################################
# MARK: conj
def _check_first_conj(node: AbstractNode) -> Optional[Successor]:
    args = (node,)
    successors = node._successors.get('conj')
    if not successors:
        return None
    return successors.get(args)


def _conj_first(node: AbstractNode) -> Node:
    new_node = Node._create_resultant(axes_names=node.axes_names,
                                      name='conj',
                                      network=node._network,
                                      tensor=node.tensor.conj(),
                                      edges=node._edges,
                                      node1_list=node.is_node1())
    
    # Create successor
    net = node._network
    args = (node,)
    successor = Successor(node_ref=node.node_ref(),
                          index=node._tensor_info['index'],
                          child=new_node)

    # Add successor to parent
    if 'conj' in node._successors:
        node._successors['conj'].update({args: successor})
    else:
        node._successors['conj'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('conj', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node._record_in_inverse_memory()

    return new_node


def _conj_next(successor: Successor, node: AbstractNode) -> Node:
    tensor = node._direct_get_tensor(successor.node_ref,
                                     successor.index)
    child = successor.child
    child._direct_set_tensor(tensor.conj())

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node._network._traced:
        node._check_inverse_memory(successor.node_ref)
    
    return child


conj_op = Operation('conj',
                    _check_first_conj,
                    _conj_first,
                    _conj_next)


def conj(node: AbstractNode) -> Node:
    """
    Returns a view of the node's tensor with a flipped conjugate bit. If the
    node has a non-complex dtype, this function returns a new node with the
    same tensor.

    See `conj <https://pytorch.org/docs/stable/generated/torch.conj.html>`_
    in the **PyTorch** documentation.

    Parameters
    ----------
    node : Node or ParamNode
        Node that is to be conjugated.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn((3, 3), dtype=torch.complex64)
    >>> conjA = tk.conj(nodeA)
    >>> conjA.is_conj()
    True
    """
    return conj_op(node)


conj_node = copy_func(conj)
conj_node.__doc__ = \
    """
    Returns a view of the node's tensor with a flipped conjugate bit. If the
    node has a non-complex dtype, this function returns a new node with the
    same tensor.

    See `conj <https://pytorch.org/docs/stable/generated/torch.conj.html>`_
    in the **PyTorch** documentation.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn((3, 3), dtype=torch.complex64)
    >>> conjA = nodeA.conj()
    >>> conjA.is_conj()
    True
    """

AbstractNode.conj = conj_node


###############################################################################
#                            NODE-LIKE OPERATIONS                             #
###############################################################################

##################################   SPLIT    #################################
# MARK: split
def _check_first_split(node: AbstractNode,
                       node1_axes: Sequence[Ax],
                       node2_axes: Sequence[Ax],
                       mode: Text = 'svd',
                       side: Optional[Text] = 'left',
                       rank: Optional[int] = None,
                       cum_percentage: Optional[float] = None,
                       cutoff: Optional[float] = None) -> Optional[Successor]:
    args = (node,
            tuple(node1_axes),
            tuple(node2_axes),
            mode,
            side,
            rank,
            cum_percentage,
            cutoff)
    successors = node._successors.get('split')
    if not successors:
        return None
    return successors.get(args)


def _split_first(node: AbstractNode,
                 node1_axes: Sequence[Ax],
                 node2_axes: Sequence[Ax],
                 mode: Text = 'svd',
                 side: Optional[Text] = 'left',
                 rank: Optional[int] = None,
                 cum_percentage: Optional[float] = None,
                 cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    if not isinstance(node1_axes, Sequence):
        raise TypeError('`node1_edges` should be list or tuple type')
    if not isinstance(node2_axes, Sequence):
        raise TypeError('`node2_edges` should be list or tuple type')
    
    args = (node,
            tuple(node1_axes),
            tuple(node2_axes),
            mode,
            side,
            rank,
            cum_percentage,
            cutoff)

    node1_axes = [node.get_axis_num(axis) for axis in node1_axes]
    node2_axes = [node.get_axis_num(axis) for axis in node2_axes]

    batch_axes = []
    all_axes = node1_axes + node2_axes
    all_axes.sort()

    if all_axes:
        j = 0
        k = all_axes[0]
    else:
        k = node.rank

    for i in range(node.rank):
        if i < k:
            if not node._edges[i].is_batch():
                raise ValueError(f'Edge {node._edges[i]} is not a batch '
                                 'edge but it\'s not included in `node1_axes` '
                                 'neither in `node2_axes`')
            else:
                batch_axes.append(i)
        else:
            if (j + 1) == len(all_axes):
                k = node.rank
            else:
                j += 1
                k = all_axes[j]

    batch_shape = torch.tensor(node.shape)[batch_axes].tolist()
    node1_shape = torch.tensor(node.shape)[node1_axes]
    node2_shape = torch.tensor(node.shape)[node2_axes]

    permutation_dims = batch_axes + node1_axes + node2_axes
    if permutation_dims == list(range(node.rank)):
        permutation_dims = []

    if permutation_dims:
        node_tensor = node.tensor \
            .permute(*(batch_axes + node1_axes + node2_axes)) \
            .reshape(*(batch_shape +
                       [node1_shape.prod().item()] +
                       [node2_shape.prod().item()]))
    else:
        node_tensor = node.tensor \
            .reshape(*(batch_shape +
                       [node1_shape.prod().item()] +
                       [node2_shape.prod().item()]))

    if (mode == 'svd') or (mode == 'svdr'):
        u, s, vh = torch.linalg.svd(node_tensor, full_matrices=False)
        
        lst_ranks = []
        
        if rank is None:
            rank = s.shape[-1]
            lst_ranks.append(rank)
        else:
            lst_ranks.append(min(max(1, int(rank)), s.shape[-1]))
            
        if cum_percentage is not None:
            s_percentages = s.cumsum(-1) / \
                (s.sum(-1, keepdim=True).expand(s.shape) + 1e-10) # To avoid having all 0's
            cum_percentage_tensor = cum_percentage * torch.ones_like(s)
            cp_rank = torch.lt(
                s_percentages,
                cum_percentage_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, cp_rank.item() + 1))
            
        if cutoff is not None:
            cutoff_tensor = cutoff * torch.ones_like(s)
            co_rank = torch.ge(
                s,
                cutoff_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, co_rank.item()))
        
        # Select rank from specified restrictions
        rank = min(lst_ranks)
        
        u = u[..., :rank]
        s = s[..., :rank]
        vh = vh[..., :rank, :]
        
        if u.is_complex():
            s = s.to(u.dtype)

        if mode == 'svdr':
            phase = torch.sgn(torch.randn_like(s))
            phase = torch.diag_embed(phase)
            u = u @ phase
            vh = phase @ vh

        if side == 'left':
            u = u @ torch.diag_embed(s)
        elif side == 'right':
            vh = torch.diag_embed(s) @ vh
        else:
            raise ValueError('`side` can only be "left" or "right"')

        node1_tensor = u
        node2_tensor = vh

    elif mode == 'qr':
        q, r = torch.linalg.qr(node_tensor)
        rank = q.shape[-1]

        node1_tensor = q
        node2_tensor = r

    elif mode == 'rq':
        q, r = torch.linalg.qr(node_tensor.transpose(-1, -2))
        q = q.transpose(-1, -2)
        r = r.transpose(-1, -2)
        rank = r.shape[-1]

        node1_tensor = r
        node2_tensor = q

    else:
        raise ValueError('`mode` can only be "svd", "svdr", "qr" or "rq"')

    node1_tensor = node1_tensor.reshape(
        *(batch_shape + node1_shape.tolist() + [rank]))
    node2_tensor = node2_tensor.reshape(
        *(batch_shape + [rank] + node2_shape.tolist()))

    net = node._network

    node1_axes_names = permute_list(node.axes_names,
                                    batch_axes + node1_axes) + ['split']
    node1 = Node._create_resultant(axes_names=node1_axes_names,
                                   name='split',
                                   network=net,
                                   tensor=node1_tensor)

    node2_axes_names = permute_list(node.axes_names, batch_axes) + ['split'] + \
        permute_list(node.axes_names, node2_axes)
    node2 = Node._create_resultant(axes_names=node2_axes_names,
                                   name='split',
                                   network=net,
                                   tensor=node2_tensor)

    n_batches = len(batch_axes)
    for edge in node1._edges[n_batches:-1]:
        net._remove_edge(edge)
    for edge in node2._edges[(n_batches + 1):]:
        net._remove_edge(edge)

    trace_node2_axes = []
    for i, axis1 in enumerate(node1_axes):
        edge1 = node._edges[axis1]

        in_node2 = False
        for j, axis2 in enumerate(node2_axes):
            edge2 = node._edges[axis2]
            if edge1 == edge2:
                in_node2 = True
                trace_node2_axes.append(axis2)

                node1_is_node1 = node.is_node1(axis1)
                if node1_is_node1:
                    new_edge = edge1.__class__(node1=node1,
                                               axis1=n_batches + i,
                                               node2=node2,
                                               axis2=n_batches + j + 1)
                else:
                    new_edge = edge1.__class__(node1=node2,
                                               axis1=n_batches + j + 1,
                                               node2=node1,
                                               axis2=n_batches + i)

                node1._add_edge(edge=new_edge,
                                axis=n_batches + i,
                                node1=node1_is_node1)
                node2._add_edge(edge=new_edge,
                                axis=n_batches + j + 1,
                                node1=not node1_is_node1)

        if not in_node2:
            node1._add_edge(edge=edge1,
                            axis=n_batches + i,
                            node1=node.is_node1(axis1))

    for j, axis2 in enumerate(node2_axes):
        if axis2 not in trace_node2_axes:
            node2._add_edge(edge=node._edges[axis2],
                            axis=n_batches + j + 1,
                            node1=node.is_node1(axis2))

    split_edge = node1['split'] ^ node2['split']
    net._remove_edge(split_edge)

    # Create successor
    successor = Successor(node_ref=node.node_ref(),
                          index=node._tensor_info['index'],
                          child=[node1, node2],
                          hints={'batch_axes': batch_axes,
                                 'node1_axes': node1_axes,
                                 'node2_axes': node2_axes,
                                 'permutation_dims': permutation_dims})

    # Add successor to parent
    if 'split' in node._successors:
        node._successors['split'].update({args: successor})
    else:
        node._successors['split'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('split', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        node._record_in_inverse_memory()

    return node1, node2


def _split_next(successor: Successor,
                node: AbstractNode,
                node1_axes: Sequence[Ax],
                node2_axes: Sequence[Ax],
                mode: Text = 'svd',
                side: Optional[Text] = 'left',
                rank: Optional[int] = None,
                cum_percentage: Optional[float] = None,
                cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    batch_axes = successor.hints['batch_axes']
    node1_axes = successor.hints['node1_axes']
    node2_axes = successor.hints['node2_axes']
    permutation_dims = successor.hints['permutation_dims']

    batch_shape = torch.tensor(node._shape)[batch_axes].tolist()
    node1_shape = torch.tensor(node._shape)[node1_axes]
    node2_shape = torch.tensor(node._shape)[node2_axes]
    
    node_tensor = node._direct_get_tensor(successor.node_ref,
                                          successor.index)
    if permutation_dims:
        node_tensor = node_tensor \
            .permute(*(batch_axes + node1_axes + node2_axes)) \
            .reshape(*(batch_shape +
                       [node1_shape.prod().item()] +
                       [node2_shape.prod().item()]))
    else:
        node_tensor = node_tensor \
            .reshape(*(batch_shape +
                       [node1_shape.prod().item()] +
                       [node2_shape.prod().item()]))

    if (mode == 'svd') or (mode == 'svdr'):
        u, s, vh = torch.linalg.svd(node_tensor, full_matrices=False)
        
        lst_ranks = []
        
        if rank is None:
            rank = s.shape[-1]
            lst_ranks.append(rank)
        else:
            lst_ranks.append(min(max(1, rank), s.shape[-1]))
            
        if cum_percentage is not None:
            s_percentages = s.cumsum(-1) / \
                (s.sum(-1, keepdim=True).expand(s.shape) + 1e-10) # To avoid having all 0's
            cum_percentage_tensor = cum_percentage * torch.ones_like(s)
            cp_rank = torch.lt(
                s_percentages,
                cum_percentage_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, cp_rank.item() + 1))
            
        if cutoff is not None:
            cutoff_tensor = cutoff * torch.ones_like(s)
            co_rank = torch.ge(
                s,
                cutoff_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, co_rank.item()))
        
        # Select rank from specified restrictions
        rank = min(lst_ranks)
        
        u = u[..., :rank]
        s = s[..., :rank]
        vh = vh[..., :rank, :]
        
        if u.is_complex():
            s = s.to(u.dtype)

        if mode == 'svdr':
            phase = torch.sgn(torch.randn_like(s))
            phase = torch.diag_embed(phase)
            u = u @ phase
            vh = phase @ vh

        if side == 'left':
            u = u @ torch.diag_embed(s)
        elif side == 'right':
            vh = torch.diag_embed(s) @ vh
        else:
            raise ValueError('`side` can only be "left" or "right"')

        node1_tensor = u
        node2_tensor = vh

    elif mode == 'qr':
        q, r = torch.linalg.qr(node_tensor)
        rank = q.shape[-1]

        node1_tensor = q
        node2_tensor = r

    elif mode == 'rq':
        q, r = torch.linalg.qr(node_tensor.transpose(-1, -2))
        q = q.transpose(-1, -2)
        r = r.transpose(-1, -2)
        rank = r.shape[-1]

        node1_tensor = r
        node2_tensor = q

    else:
        raise ValueError('`mode` can only be "svd", "svdr", "qr" or "rq"')

    node1_tensor = node1_tensor.reshape(
        *(batch_shape + node1_shape.tolist() + [rank]))
    node2_tensor = node2_tensor.reshape(
        *(batch_shape + [rank] + node2_shape.tolist()))

    children = successor.child
    children[0]._direct_set_tensor(node1_tensor)
    children[1]._direct_set_tensor(node2_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if node._network._traced:
        node._check_inverse_memory(successor.node_ref)

    return children[0], children[1]


split_op = Operation('split', _check_first_split, _split_first, _split_next)


def split(node: AbstractNode,
          node1_axes: Sequence[Ax],
          node2_axes: Sequence[Ax],
          mode: Text = 'svd',
          side: Optional[Text] = 'left',
          rank: Optional[int] = None,
          cum_percentage: Optional[float] = None,
          cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    Splits one node in two via the decomposition specified in ``mode``. To
    perform this operation the set of edges has to be split in two sets,
    corresponding to the edges of the first and second ``resultant`` nodes.
    Batch edges that don't appear in any of the lists will be repeated in both
    nodes, and will appear as the first edges of the ``resultant`` nodes, in
    the order they appeared in ``node``.

    Having specified the two sets of edges, the node's tensor is reshaped as a
    batch matrix, with batch dimensions first, a single input dimension
    (adding up all edges in the first set) and a single output dimension
    (adding up all edges in the second set). With this shape, each matrix in
    the batch is decomposed according to ``mode``.

    * **"svd"**: Singular Value Decomposition

      .. math::

        M = USV^{\dagger}

      where :math:`U` and :math:`V` are unitary, and :math:`S` is diagonal.

    * **"svdr"**: Singular Value Decomposition adding Random phases (square
      diagonal matrices with random 1's and -1's)

      .. math::

        M = UR_1SR_2V^{\dagger}

      where :math:`U` and :math:`V` are unitary, :math:`S` is diagonal, and
      :math:`R_1` and :math:`R_2` are square diagonal matrices with random 1's
      and -1's.

    * **"qr"**: QR decomposition

      .. math::

        M = QR

      where Q is unitary and R is an upper triangular matrix.

    * **"rq"**: RQ decomposition

      .. math::

        M = RQ

      where R is a lower triangular matrix and Q is unitary.

    If ``mode`` is "svd" or "svdr", ``side`` must be provided. Besides, at least
    one of ``rank``, ``cum_percentage`` and ``cutoff`` is required. If more than
    one is specified, the resulting rank will be the one that satisfies all
    conditions.
    
    Since the node is `split` in two, a new edge appears connecting both
    nodes. The axis that corresponds to this edge has the name ``"split"``.
    
    Nodes ``resultant`` from this operation are called ``"split"``. The node
    that keeps information about the :class:`Successor` is ``node``.
    
    This operation is the same as :meth:`~AbstractNode.split`.

    Parameters
    ----------
    node : AbstractNode
        Node that is to be split.
    node1_axes : list[int, str or Axis]
        First set of edges, will appear as the edges of the first (left)
        resultant node.
    node2_axes : list[int, str or Axis]
        Second set of edges, will appear as the edges of the second (right)
        resultant node.
    mode : {"svd", "svdr", "qr", "rq"}
        Decomposition to be used.
    side : str, optional
        If ``mode`` is "svd" or "svdr", indicates the side to which the diagonal
        matrix :math:`S` should be contracted. If "left", the first resultant
        node's tensor will be :math:`US`, and the other node's tensor will be
        :math:`V^{\dagger}`. If "right", their tensors will be :math:`U` and
        :math:`SV^{\dagger}`, respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> node = tk.randn(shape=(10, 15, 100),
    ...                 axes_names=('left', 'right', 'batch'))
    >>> node_left, node_right = tk.split(node,
    ...                                  ['left'], ['right'],
    ...                                  mode='svd',
    ...                                  rank=5)
    >>> node_left.shape
    torch.Size([100, 10, 5])
    
    >>> node_right.shape
    torch.Size([100, 5, 15])
    
    >>> node_left['split']
    Edge( split_0[split] <-> split_1[split] )
    """
    return split_op(node, node1_axes, node2_axes,
                    mode, side, rank, cum_percentage, cutoff)


split_node = copy_func(split)
split_node.__doc__ = \
    r"""
    Splits one node in two via the decomposition specified in ``mode``. See
    :func:`split` for a more complete explanation.
    
    Since the node is `split` in two, a new edge appears connecting both
    nodes. The axis that corresponds to this edge has the name ``"split"``.
    
    Nodes ``resultant`` from this operation are called ``"split"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node1_axes : list[int, str or Axis]
        First set of edges, will appear as the edges of the first (left)
        resultant node.
    node2_axes : list[int, str or Axis]
        Second set of edges, will appear as the edges of the second (right)
        resultant node.
    mode : {"svd", "svdr", "qr", "rq"}
        Decomposition to be used.
    side : str, optional
        If ``mode`` is "svd" or "svdr", indicates the side to which the diagonal
        matrix :math:`S` should be contracted. If "left", the first resultant
        node's tensor will be :math:`US`, and the other node's tensor will be
        :math:`V^{\dagger}`. If "right", their tensors will be :math:`U` and
        :math:`SV^{\dagger}`, respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.
        
        .. math::
        
            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> node = tk.randn(shape=(10, 15, 100),
    ...                 axes_names=('left', 'right', 'batch'))
    >>> node_left, node_right = node.split(['left'], ['right'],
    ...                                    mode='svd',
    ...                                    rank=5)
    >>> node_left.shape
    torch.Size([100, 10, 5])
    
    >>> node_right.shape
    torch.Size([100, 5, 15])
    
    >>> node_left['split']
    Edge( split_0[split] <-> split_1[split] )
    """

AbstractNode.split = split_node


def split_(node: AbstractNode,
           node1_axes: Sequence[Ax],
           node2_axes: Sequence[Ax],
           mode: Text = 'svd',
           side: Optional[Text] = 'left',
           rank: Optional[int] = None,
           cum_percentage: Optional[float] = None,
           cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    In-place version of :func:`split`.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Since the node is `split` in two, a new edge appears connecting both
    nodes. The axis that corresponds to this edge has the name ``"split"``.
    
    Nodes ``resultant`` from this operation are called ``"split_ip"``.
    
    This operation is the same as :meth:`~AbstractNode.split_`.

    Parameters
    ----------
    node : AbstractNode
        Node that is to be split.
    node1_axes : list[int, str or Axis]
        First set of edges, will appear as the edges of the first (left)
        resultant node.
    node2_axes : list[int, str or Axis]
        Second set of edges, will appear as the edges of the second (right)
        resultant node.
    mode : {"svd", "svdr", "qr", "rq"}
        Decomposition to be used.
    side : str, optional
        If ``mode`` is "svd" or "svdr", indicates the side to which the diagonal
        matrix :math:`S` should be contracted. If "left", the first resultant
        node's tensor will be :math:`US`, and the other node's tensor will be
        :math:`V^{\dagger}`. If "right", their tensors will be :math:`U` and
        :math:`SV^{\dagger}`, respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> node = tk.randn(shape=(10, 15, 100),
    ...                 axes_names=('left', 'right', 'batch'))
    >>> node_left, node_right = tk.split_(node,
    ...                                   ['left'], ['right'],
    ...                                   mode='svd',
    ...                                   rank=5)
    >>> node_left.shape
    torch.Size([100, 10, 5])
    
    >>> node_right.shape
    torch.Size([100, 5, 15])
    
    >>> node_left['split']
    Edge( split_ip_0[split] <-> split_ip_1[split] )
    
    ``node`` has been deleted (removed from the network), but it still exists
    until is deleted.
    
    >>> node.network is None
    True
    
    >>> del node
    """
    node1, node2 = split(node, node1_axes, node2_axes,
                         mode, side, rank, cum_percentage, cutoff)
    node1.reattach_edges(override=True)
    node2.reattach_edges(override=True)
    node1._unrestricted_set_tensor(node1.tensor.detach())
    node2._unrestricted_set_tensor(node2.tensor.detach())

    # Delete node (and its edges) from the TN
    net = node._network
    net.delete_node(node)

    # Add edges of result to the TN
    for res_edge in node1._edges + node2._edges:
        net._add_edge(res_edge)

    # Transform resultant to leaf nodes
    node1._leaf = True
    del net._resultant_nodes[node1._name]
    net._leaf_nodes[node1._name] = node1

    node2._leaf = True
    del net._resultant_nodes[node2._name]
    net._leaf_nodes[node2._name] = node2

    node._successors = dict()
    net._seq_ops = []

    # Remove resultant names
    node1.name = 'split_ip'
    node2.name = 'split_ip'

    return node1, node2


split_node_ = copy_func(split_)
split_node_.__doc__ = \
    r"""
    In-place version of :meth:`~AbstractNode.split`.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Since the node is `split` in two, a new edge appears connecting both
    nodes. The axis that corresponds to this edge has the name ``"split"``.
    
    Nodes ``resultant`` from this operation are called ``"split_ip"``.

    Parameters
    ----------
    node1_axes : list[int, str or Axis]
        First set of edges, will appear as the edges of the first (left)
        resultant node.
    node2_axes : list[int, str or Axis]
        Second set of edges, will appear as the edges of the second (right)
        resultant node.
    mode : {"svd", "svdr", "qr", "rq"}
        Decomposition to be used.
    side : str, optional
        If ``mode`` is "svd" or "svdr", indicates the side to which the diagonal
        matrix :math:`S` should be contracted. If "left", the first resultant
        node's tensor will be :math:`US`, and the other node's tensor will be
        :math:`V^{\dagger}`. If "right", their tensors will be :math:`U` and
        :math:`SV^{\dagger}`, respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.
        
        .. math::
        
            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> node = tk.randn(shape=(10, 15, 100),
    ...                 axes_names=('left', 'right', 'batch'))
    >>> node_left, node_right = node.split_(['left'], ['right'],
    ...                                     mode='svd',
    ...                                     rank=5)
    >>> node_left.shape
    torch.Size([100, 10, 5])
    
    >>> node_right.shape
    torch.Size([100, 5, 15])
    
    >>> node_left['split']
    Edge( split_ip_0[split] <-> split_ip_1[split] )
    
    ``node`` has been deleted (removed from the network), but it still exists
    until is deleted.
    
    >>> node.network is None
    True
    
    >>> del node
    """

AbstractNode.split_ = split_node_


def svd(edge: Edge,
        side: Text = 'left',
        rank: Optional[int] = None,
        cum_percentage: Optional[float] = None,
        cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    Contracts an edge via :func:`contract` and splits it via :func:`split`
    using ``mode = "svd"``. See :func:`split` for a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    This operation is the same as :meth:`~Edge.svd`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = tk.svd(new_edge, rank=7)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> new_nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract()
    new_node1, new_node2 = split(node=contracted,
                                 node1_axes=list(
                                     range(n_batches,
                                         n_batches + n_axes1)),
                                 node2_axes=list(
                                     range(n_batches + n_axes1,
                                         n_batches + n_axes1 + n_axes2)),
                                 mode='svd',
                                 side=side,
                                 rank=rank,
                                 cum_percentage=cum_percentage,
                                 cutoff=cutoff)

    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute(permutation)
        
    new_node1.get_axis(axis1._num).name = axis1._name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


svd_edge = copy_func(svd)
svd_edge.__doc__ = \
    r"""
    Contracts an edge via :meth:`~Edge.contract` and splits it via
    :meth:`~AbstractNode.split` using ``mode = "svd"``. See :func:`split` for
    a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.

    Parameters
    ----------
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = new_edge.svd(rank=7)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> new_nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """

Edge.svd = svd_edge


def svd_(edge: Edge,
         side: Text = 'left',
         rank: Optional[int] = None,
         cum_percentage: Optional[float] = None,
         cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    In-place version of :func:`svd`.
    
    Contracts an edge in-place via :func:`contract_` and splits it in-place via
    :func:`split_` using ``mode = "svd"``. See :func:`split` for a more complete
    explanation.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``edge``.
    
    This operation is the same as :meth:`~Edge.svd_`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = tk.svd_(new_edge, rank=7)
    ...
    >>> nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1._name, node2._name
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(
                                      range(n_batches,
                                            n_batches + n_axes1)),
                                  node2_axes=list(
                                      range(n_batches + n_axes1,
                                            n_batches + n_axes1 + n_axes2)),
                                  mode='svd',
                                  side=side,
                                  rank=rank,
                                  cum_percentage=cum_percentage,
                                  cutoff=cutoff)

    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)

    new_node1.name = node1_name
    new_node1.get_axis(axis1._num).name = axis1._name

    new_node2.name = node2_name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


svd_edge_ = copy_func(svd_)
svd_edge_.__doc__ = \
    r"""
    In-place version of :meth:`~Edge.svd`.
    
    Contracts an edge in-place via :meth:`~Edge.contract_` and splits
    it in-place via :meth:`~AbstractNode.split_` using ``mode = "svd"``. See
    :func:`split` for a more complete explanation.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``self``.

    Parameters
    ----------
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.
        
        .. math::
        
            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = new_edge.svd_(rank=7)
    ...
    >>> nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """

Edge.svd_ = svd_edge_


def svdr(edge: Edge,
         side: Text = 'left',
         rank: Optional[int] = None,
         cum_percentage: Optional[float] = None,
         cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    Contracts an edge via :func:`contract` and splits it via :func:`split`
    using ``mode = "svdr"``. See :func:`split` for a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    This operation is the same as :meth:`~Edge.svdr`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = tk.svdr(new_edge, rank=7)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> new_nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract()
    new_node1, new_node2 = split(node=contracted,
                                 node1_axes=list(
                                     range(n_batches,
                                         n_batches + n_axes1)),
                                 node2_axes=list(
                                     range(n_batches + n_axes1,
                                         n_batches + n_axes1 + n_axes2)),
                                 mode='svdr',
                                 side=side,
                                 rank=rank,
                                 cum_percentage=cum_percentage,
                                 cutoff=cutoff)

    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute(permutation)
        
    new_node1.get_axis(axis1._num).name = axis1._name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


svdr_edge = copy_func(svdr)
svdr_edge.__doc__ = \
    r"""
    Contracts an edge via :meth:`~Edge.contract` and splits it via
    :meth:`~AbstractNode.split` using ``mode = "svdr"``. See :func:`split` for
    a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.

    Parameters
    ----------
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = new_edge.svdr(rank=7)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> new_nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """

Edge.svdr = svdr_edge


def svdr_(edge: Edge,
          side: Text = 'left',
          rank: Optional[int] = None,
          cum_percentage: Optional[float] = None,
          cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    r"""
    In-place version of :func:`svdr`.
    
    Contracts an edge in-place via :func:`contract_` and splits it in-place via
    :func:`split_` using ``mode = "svdr"``. See :func:`split` for a more complete
    explanation.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``edge``.
    
    This operation is the same as :meth:`~Edge.svdr_`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = tk.svdr_(new_edge, rank=7)
    ...
    >>> nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1._name, node2._name
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(
                                      range(n_batches,
                                            n_batches + n_axes1)),
                                  node2_axes=list(
                                      range(n_batches + n_axes1,
                                            n_batches + n_axes1 + n_axes2)),
                                  mode='svdr',
                                  side=side,
                                  rank=rank,
                                  cum_percentage=cum_percentage,
                                  cutoff=cutoff)

    # new_node1
    prev_nums = [ax._num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)

    new_node1.name = node1_name
    new_node1.get_axis(axis1._num).name = axis1._name

    new_node2.name = node2_name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


svdr_edge_ = copy_func(svdr_)
svdr_edge_.__doc__ = \
    r"""
    In-place version of :meth:`~Edge.svdr`.
    
    Contracts an edge in-place via :meth:`~Edge.contract_` and splits
    it in-place via :meth:`~AbstractNode.split_` using ``mode = "svdr"``. See
    :func:`split` for a more complete explanation.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``self``.

    Parameters
    ----------
    side : str, optional
        Indicates the side to which the diagonal matrix :math:`S` should be
        contracted. If "left", the first resultant node's tensor will be
        :math:`US`, and the other node's tensor will be :math:`V^{\dagger}`.
        If "right", their tensors will be :math:`U` and :math:`SV^{\dagger}`,
        respectively.
    rank : int, optional
        Number of singular values to keep.
    cum_percentage : float, optional
        Proportion that should be satisfied between the sum of all singular
        values kept and the total sum of all singular values.
        
        .. math::
        
            \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
            cum\_percentage
    cutoff : float, optional
        Quantity that lower bounds singular values in order to be kept.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = new_edge.svdr_(rank=7)
    ...
    >>> nodeA.shape
    torch.Size([10, 7, 100])
    
    >>> nodeB.shape
    torch.Size([7, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """

Edge.svdr_ = svdr_edge_


def qr(edge: Edge) -> Tuple[Node, Node]:
    r"""
    Contracts an edge via :func:`contract` and splits it via :func:`split`
    using ``mode = "qr"``. See :func:`split` for a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    This operation is the same as :meth:`~Edge.qr`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = tk.qr(new_edge)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> new_nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract()
    new_node1, new_node2 = split(node=contracted,
                                 node1_axes=list(
                                     range(n_batches,
                                         n_batches + n_axes1)),
                                 node2_axes=list(
                                     range(n_batches + n_axes1,
                                         n_batches + n_axes1 + n_axes2)),
                                 mode='qr')

    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute(permutation)
        
    new_node1.get_axis(axis1._num).name = axis1._name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


qr_edge = copy_func(qr)
qr_edge.__doc__ = \
    r"""
    Contracts an edge via :meth:`~Edge.contract` and splits it via
    :meth:`~AbstractNode.split` using ``mode = "qr"``. See :func:`split` for
    a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = new_edge.qr()
    ...
    >>> new_nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> new_nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """

Edge.qr = qr_edge


def qr_(edge) -> Tuple[Node, Node]:
    r"""
    In-place version of :func:`qr`.
    
    Contracts an edge in-place via :func:`contract_` and splits it in-place via
    :func:`split_` using ``mode = "qr"``. See :func:`split` for a more complete
    explanation.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``edge``.
    
    This operation is the same as :meth:`~Edge.qr_`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = tk.qr_(new_edge)
    ...
    >>> nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1._name, node2._name
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(
                                      range(n_batches,
                                            n_batches + n_axes1)),
                                  node2_axes=list(
                                      range(n_batches + n_axes1,
                                            n_batches + n_axes1 + n_axes2)),
                                  mode='qr')

    # new_node1
    prev_nums = [ax._num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)

    new_node1.name = node1_name
    new_node1.get_axis(axis1._num).name = axis1._name

    new_node2.name = node2_name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


qr_edge_ = copy_func(qr_)
qr_edge_.__doc__ = \
    r"""
    In-place version of :meth:`~Edge.qr`.
    
    Contracts an edge in-place via :meth:`~Edge.contract_` and splits
    it in-place via :meth:`~AbstractNode.split_` using ``mode = "qr"``. See
    :func:`split` for a more complete explanation.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``self``.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = new_edge.qr_()
    ...
    >>> nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """

Edge.qr_ = qr_edge_


def rq(edge: Edge) -> Tuple[Node, Node]:
    r"""
    Contracts an edge via :func:`contract` and splits it via :func:`split`
    using ``mode = "rq"``. See :func:`split` for a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    This operation is the same as :meth:`~Edge.rq`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = tk.rq(new_edge)
    ...
    >>> new_nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> new_nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract()
    new_node1, new_node2 = split(node=contracted,
                                 node1_axes=list(
                                     range(n_batches,
                                         n_batches + n_axes1)),
                                 node2_axes=list(
                                     range(n_batches + n_axes1,
                                         n_batches + n_axes1 + n_axes2)),
                                 mode='rq')

    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute(permutation)
        
    new_node1.get_axis(axis1._num).name = axis1._name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


rq_edge = copy_func(rq)
rq_edge.__doc__ = \
    r"""
    Contracts an edge via :meth:`~Edge.contract` and splits it via
    :meth:`~AbstractNode.split` using ``mode = "rq"``. See :func:`split` for
    a more complete explanation.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> new_nodeA, new_nodeB = new_edge.rq()
    ...
    >>> new_nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> new_nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(new_nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(new_nodeB.axes_names)
    ['left', 'right', 'batch']
    
    Original nodes still exist in the network
    
    >>> assert nodeA.network == new_nodeA.network
    >>> assert nodeB.network == new_nodeB.network
    """

Edge.rq = rq_edge


def rq_(edge) -> Tuple[Node, Node]:
    r"""
    In-place version of :func:`rq`.
    
    Contracts an edge in-place via :func:`contract_` and splits it in-place via
    :func:`split_` using ``mode = "rq"``. See :func:`split` for a more complete
    explanation.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``edge``.
    
    This operation is the same as :meth:`~Edge.rq_`.

    Parameters
    ----------
    edge : Edge
        Edge whose nodes are to be contracted and split.

    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = tk.rq_(new_edge)
    ...
    >>> nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    if edge.node1 is edge.node2:
        raise ValueError('Edge should connect different nodes')

    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1._name, node2._name
    axis1, axis2 = edge.axis1, edge.axis2

    batch_axes = []
    for axis in node1._axes:
        if axis._batch and (axis._name in node2.axes_names):
            batch_axes.append(axis)

    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1

    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(
                                      range(n_batches,
                                            n_batches + n_axes1)),
                                  node2_axes=list(
                                      range(n_batches + n_axes1,
                                            n_batches + n_axes1 + n_axes2)),
                                  mode='rq')

    # new_node1
    prev_nums = [ax._num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1._num):
            prev_nums.append(i)
    prev_nums += [axis1._num]

    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)

    # new_node2
    prev_nums = [node2.get_axis_num(node1.get_axis(ax)._name)
                 for ax in batch_axes] + [axis2._num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)

    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)

    new_node1.name = node1_name
    new_node1.get_axis(axis1._num).name = axis1._name

    new_node2.name = node2_name
    new_node2.get_axis(axis2._num).name = axis2._name

    return new_node1, new_node2


rq_edge_ = copy_func(rq_)
rq_edge_.__doc__ = \
    r"""
    In-place version of :meth:`~Edge.rq`.
    
    Contracts an edge in-place via :meth:`~Edge.contract_` and splits
    it in-place via :meth:`~AbstractNode.split_` using ``mode = "qr"``. See
    :func:`split` for a more complete explanation.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation use the same names as the original
    nodes connected by ``self``.
    
    Returns
    -------
    tuple[Node, Node]
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 20, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA, nodeB = tk.rq_(new_edge)
    ...
    >>> nodeA.shape
    torch.Size([10, 10, 100])
    
    >>> nodeB.shape
    torch.Size([10, 20, 100])
    
    >>> print(nodeA.axes_names)
    ['left', 'right', 'batch']
    
    >>> print(nodeB.axes_names)
    ['left', 'right', 'batch']
    """

Edge.rq_ = rq_edge_


################################   CONTRACT    ################################
# MARK: contract_edges
def _check_first_contract_edges(edges: Optional[List[Edge]],
                                node1: AbstractNode,
                                node2: AbstractNode) -> Optional[Successor]:
    args = (None if edges is None else tuple(edges), node1, node2)
    successors = node1._successors.get('contract_edges')
    if successors is None:
        return None
    return successors.get(args)


def _permute_contract_reshape(tensor1, tensor2, permutation_dims, shape_limits):
    batch = shape_limits[0]
    non_contract_0 = shape_limits[1]
    contract = shape_limits[2]
    
    # Permute if needed
    permute1 = tensor1
    if len(permutation_dims[0]) > 0:
        permute1 = tensor1.permute(permutation_dims[0])
        
    permute2 = tensor2
    if len(permutation_dims[1]) > 0:
        permute2 = tensor2.permute(permutation_dims[1])
        
    # Compute sizes for reshapes
    aux_shape1 = [permute1.shape[:batch].numel(),
                  permute1.shape[batch:(batch + non_contract_0)].numel(),
                  permute1.shape[(batch + non_contract_0):].numel()]
    aux_shape2 = [permute2.shape[:batch].numel(),
                  permute2.shape[batch:(batch + contract)].numel(),
                  permute2.shape[(batch + contract):].numel(),]
    new_shape = \
        list(permute1.shape[:batch]) + \
        list(permute1.shape[batch:(batch + non_contract_0)]) + \
        list(permute2.shape[(batch + contract):])
    
    # Reshape
    reshape1 = permute1.reshape(aux_shape1)
    reshape2 = permute2.reshape(aux_shape2)
    
    # Contract and reshape
    result = torch.bmm(reshape1, reshape2)
    result = result.reshape(new_shape)
    
    return result


def _contract_edges_first(edges: Optional[List[Edge]],
                          node1: AbstractNode,
                          node2: AbstractNode) -> Node:
    shared_edges = get_shared_edges(node1, node2)
    if not shared_edges:
        raise ValueError(f'No batch edges or shared edges between nodes '
                         f'{node1!s} and {node2!s} found')
        
    args = (None if edges is None else tuple(edges), node1, node2)

    if edges is None:
        edges = shared_edges
    else:
        for edge in edges:
            if edge not in shared_edges:
                raise ValueError('Edges selected to be contracted must be '
                                 'shared edges between `node1` and `node2`')

    # Trace
    if node1 == node2:
        result = node1.tensor
        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))

        for edge in edges:
            axes = node1.in_which_axis(edge)
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[axes[0]._num],
                                    dim2=axes_nums[axes[1]._num])
            result = result.sum(-1)
            min_axis = min(axes[0]._num, axes[1]._num)
            max_axis = max(axes[0]._num, axes[1]._num)
            for num in axes_nums:
                if num < min_axis:
                    continue
                elif num == min_axis:
                    axes_nums[num] = -1
                elif (num > min_axis) and (num < max_axis):
                    axes_nums[num] -= 1
                elif num == max_axis:
                    axes_nums[num] = -1
                elif num > max_axis:
                    axes_nums[num] -= 2

        new_axes_names = []
        new_edges = []
        new_node1_list = []
        for num in axes_nums:
            if axes_nums[num] >= 0:
                new_axes_names.append(node1._axes[num]._name)
                new_edges.append(node1._edges[num])
                new_node1_list.append(node1.is_node1(num))

        hints = edges

        # Record in inverse_memory while tracing
        if node1._network._tracing:
            node1._record_in_inverse_memory()

    else:
        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]
        non_contract_edges = [dict(), dict()]
        batch_edges = dict()
        contract_edges = dict()

        for i in [0, 1]:
            for j, axis in enumerate(nodes[i]._axes):
                edge = nodes[i]._edges[j]
                if edge in edges:
                    if i == 0:
                        contract_edges[edge] = [j]
                    else:
                        contract_edges[edge].append(j)

                elif axis._batch:
                    if i == 0:
                        batch_in_node2 = False
                        for aux_axis in nodes[1]._axes:
                            if aux_axis._batch and \
                                    (axis._name == aux_axis._name):
                                batch_edges[axis._name] = [j]
                                batch_in_node2 = True
                                break
                        if not batch_in_node2:
                            non_contract_edges[i][axis._name] = j
                    else:
                        if axis._name in batch_edges:
                            batch_edges[axis._name].append(j)
                        else:
                            non_contract_edges[i][axis._name] = j
                else:
                    non_contract_edges[i][axis._name] = j

        permutation_dims = [None, None]

        batch_edges_perm_0 = list(map(lambda l: l[0], batch_edges.values()))
        batch_edges_perm_1 = list(map(lambda l: l[1], batch_edges.values()))

        non_contract_edges_perm_0 = list(non_contract_edges[0].values())
        non_contract_edges_perm_1 = list(non_contract_edges[1].values())

        contract_edges_perm_0 = list(
            map(lambda l: l[0], contract_edges.values()))
        contract_edges_perm_1 = list(
            map(lambda l: l[1], contract_edges.values()))

        permutation_dims[0] = batch_edges_perm_0 + non_contract_edges_perm_0 + \
            contract_edges_perm_0
        permutation_dims[1] = batch_edges_perm_1 + contract_edges_perm_1 + \
            non_contract_edges_perm_1

        for i in [0, 1]:
            if permutation_dims[i] == list(range(len(permutation_dims[i]))):
                permutation_dims[i] = []

        shape_limits = (len(batch_edges),
                        len(non_contract_edges[0]),
                        len(contract_edges))
        
        result = _permute_contract_reshape(tensors[0], tensors[1],
                                           permutation_dims,
                                           shape_limits)

        # Put batch dims at the beginning
        indices = [None, None]
        indices[0] = list(map(lambda l: l[0], batch_edges.values())) + \
            list(non_contract_edges[0].values())
        indices[1] = list(non_contract_edges[1].values())

        new_axes_names = []
        new_edges = []
        new_node1_list = []
        for i in [0, 1]:
            for idx in indices[i]:
                new_axes_names.append(nodes[i].axes_names[idx])
                new_edges.append(nodes[i][idx])
                new_node1_list.append(nodes[i].axes[idx].is_node1())

        hints = {'shape_limits': shape_limits,
                 'permutation_dims': permutation_dims}

        # Record in inverse_memory while tracing
        if node1._network._tracing:
            node1._record_in_inverse_memory()
            node2._record_in_inverse_memory()

    node1_is_stack = isinstance(node1, (StackNode, ParamStackNode))
    node2_is_stack = isinstance(node2, (StackNode, ParamStackNode))
    if node1_is_stack and node2_is_stack:
        new_node = StackNode._create_resultant(axes_names=new_axes_names,
                                               name='contract_edges',
                                               network=node1._network,
                                               tensor=result,
                                               edges=new_edges,
                                               node1_list=new_node1_list)
    elif node1_is_stack or node2_is_stack:
        raise TypeError('Can only contract (Param)StackNode with other '
                        '(Param)StackNode')
    else:
        new_node = Node._create_resultant(axes_names=new_axes_names,
                                          name='contract_edges',
                                          network=node1._network,
                                          tensor=result,
                                          edges=new_edges,
                                          node1_list=new_node1_list)

    # Create successor
    net = node1._network
    successor = Successor(node_ref=(node1.node_ref(),
                                    node2.node_ref()),
                          index=(node1._tensor_info['index'],
                                 node2._tensor_info['index']),
                          child=new_node,
                          hints=hints)

    # Add successor to parent
    if 'contract_edges' in node1._successors:
        node1._successors['contract_edges'].update({args: successor})
    else:
        node1._successors['contract_edges'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('contract_edges', args))

    return new_node


def _contract_edges_next(successor: Successor,
                         edges: Optional[List[Edge]],
                         node1: AbstractNode,
                         node2: AbstractNode) -> Node:
    if node1 == node2:
        edges = successor.hints
        result = node1._direct_get_tensor(successor.node_ref[0],
                                          successor.index[0])
        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))

        for edge in edges:
            axes = node1.in_which_axis(edge)
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[axes[0]._num],
                                    dim2=axes_nums[axes[1]._num])
            result = result.sum(-1)
            min_axis = min(axes[0]._num, axes[1]._num)
            max_axis = max(axes[0]._num, axes[1]._num)
            for num in axes_nums:
                if num < min_axis:
                    continue
                elif num == min_axis:
                    axes_nums[num] = -1
                elif (num > min_axis) and (num < max_axis):
                    axes_nums[num] -= 1
                elif num == max_axis:
                    axes_nums[num] = -1
                elif num > max_axis:
                    axes_nums[num] -= 2

        # Record in inverse_memory while contracting, if network is traced
        # (to delete memory if possible)
        if node1._network._traced:
            node1._check_inverse_memory(successor.node_ref[0])

    else:
        hints = successor.hints
        tensors = [node1._direct_get_tensor(successor.node_ref[0],
                                            successor.index[0]),
                   node2._direct_get_tensor(successor.node_ref[1],
                                            successor.index[1])]
        
        result = _permute_contract_reshape(tensors[0], tensors[1],
                                           hints['permutation_dims'],
                                           hints['shape_limits'])

        # Record in inverse_memory while contracting, if network is traced
        # (to delete memory if possible)
        if node1._network._traced:
            node1._check_inverse_memory(successor.node_ref[0])
            node2._check_inverse_memory(successor.node_ref[1])

    child = successor.child
    child._direct_set_tensor(result)

    return child


contract_edges_op = Operation('contract_edges',
                              _check_first_contract_edges,
                              _contract_edges_first,
                              _contract_edges_next)


def contract_edges(edges: Optional[List[Edge]],
                   node1: AbstractNode,
                   node2: AbstractNode) -> Node:
    """
    Contracts all selected edges between two nodes.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges"``.
    The node that keeps information about the :class:`Successor` is ``node1``.

    Parameters
    ----------
    edges : list[Edge]
        List of edges that are to be contracted. They must be edges shared
        between ``node1`` and ``node2``. Batch contraction is automatically
        performed when both nodes have batch edges with the same names.
    node1 : AbstractNode
        First node of the contraction. Its non-contracted edges will appear
        first in the list of inherited edges of the resultant node.
    node2 : AbstractNode
        Second node of the contraction. Its non-contracted edges will appear
        last in the list of inherited edges of the resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['one'] ^ nodeB['one']
    >>> _ = nodeA['two'] ^ nodeB['two']
    >>> _ = nodeA['three'] ^ nodeB['three']
    >>> result = tk.contract_edges([nodeA['one'], nodeA['three']],
    ...                            nodeA, nodeB)
    >>> result.shape
    torch.Size([15, 15])
    
    If ``node1`` and ``node2`` are the same node, the contraction is a trace.
    
    >>> result2 = tk.contract_edges([result['two_0']], result, result)
    >>> result2.shape
    torch.Size([])
    """
    return contract_edges_op(edges, node1, node2)


def contract(edge: Edge) -> Node:
    """
    Contracts the nodes that are connected through the edge.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges"``.
    The node that keeps information about the :class:`Successor` is
    ``edge.node1``.
    
    This operation is the same as :meth:`~Edge.contract`.

    Parameters
    ----------
    edge : Edge
        Edge that is to be contracted. Batch contraction is automatically
        performed when both nodes have batch edges with the same names.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['one'] ^ nodeB['one']
    >>> _ = nodeA['two'] ^ nodeB['two']
    >>> _ = nodeA['three'] ^ nodeB['three']
    >>> result = tk.contract(nodeA['one'])
    >>> result.shape
    torch.Size([15, 20, 15, 20])
    """
    return contract_edges_op([edge], edge.node1, edge.node2)

contract_edge = copy_func(contract)
contract_edge.__doc__ = \
    """
    Contracts the nodes that are connected through the edge.
    
    This only works if the nodes connected through the edge are ``leaf`` nodes.
    Otherwise, this will perform the contraction between the ``leaf`` nodes
    that were connected through this edge.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges"``.
    The node that keeps information about the :class:`Successor` is
    ``self.node1``.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['one'] ^ nodeB['one']
    >>> _ = nodeA['two'] ^ nodeB['two']
    >>> _ = nodeA['three'] ^ nodeB['three']
    >>> result = nodeA['one'].contract()
    >>> result.shape
    torch.Size([15, 20, 15, 20])
    """

Edge.contract = contract_edge


def contract_(edge: Edge) -> Node:
    """
    In-place version of :func:`contract`.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges_ip"``.
    
    This operation is the same as :meth:`~Edge.contract_`.

    Parameters
    ----------
    edge : Edge
        Edges that is to be contracted. Batch contraction is automatically
        performed when both nodes have batch edges with the same names.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['one'] ^ nodeB['one']
    >>> _ = nodeA['two'] ^ nodeB['two']
    >>> _ = nodeA['three'] ^ nodeB['three']
    >>> result = tk.contract_(nodeA['one'])
    >>> result.shape
    torch.Size([15, 20, 15, 20])
    
    ``nodeA`` and ``nodeB`` have been removed from the network.
    
    >>> nodeA.network is None
    True
    
    >>> nodeB.network is None
    True
    
    >>> del nodeA
    >>> del nodeB
    """
    nodes = [edge.node1, edge.node2]
    result = contract_edges_op([edge], nodes[0], nodes[1])
    result.reattach_edges(override=True)
    result._unrestricted_set_tensor(result.tensor.detach())
    
    nodes = set(nodes)

    # Delete nodes (and their edges) from the TN
    net = result.network
    for node in nodes:
        net.delete_node(node)

    # Add edges of result to the TN
    for res_edge in result._edges:
        net._add_edge(res_edge)

    # Transform resultant to leaf nodes
    result._leaf = True
    del net._resultant_nodes[result._name]
    net._leaf_nodes[result._name] = result
    
    for node in nodes:
        node._successors = dict()
    net._seq_ops = []

    # Remove resultant name
    result.name = 'contract_edges_ip'

    return result


contract_edge_ = copy_func(contract_)
contract_edge_.__doc__ = \
    """
    In-place version of :meth:`~Edge.contract`.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges_ip"``.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(10, 15, 20),
    ...                  axes_names=('one', 'two', 'three'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['one'] ^ nodeB['one']
    >>> _ = nodeA['two'] ^ nodeB['two']
    >>> _ = nodeA['three'] ^ nodeB['three']
    >>> result = nodeA['one'].contract_()
    >>> result.shape
    torch.Size([15, 20, 15, 20])
    
    ``nodeA`` and ``nodeB`` have been removed from the network.
    
    >>> nodeA.network is None
    True
    
    >>> nodeB.network is None
    True
    
    >>> del nodeA
    >>> del nodeB
    """

Edge.contract_ = contract_edge_


def get_shared_edges(node1: AbstractNode,
                     node2: AbstractNode) -> List[Edge]:
    """Returns list of edges shared between two nodes."""
    edges = set()
    for i1, edge1 in enumerate(node1._edges):
        for i2, edge2 in enumerate(node2._edges):
            if (edge1 == edge2) and not edge1.is_dangling():
                if node1.is_node1(i1) != node2.is_node1(i2):
                    edges.add(edge1)

    return list(edges)


def contract_between(node1: AbstractNode,
                     node2: AbstractNode) -> Node:
    """
    Contracts all edges shared between two nodes. Batch contraction is
    automatically performed when both nodes have batch edges with the same
    names. It can also be performed using the operator ``@``.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges"``.
    The node that keeps information about the :class:`Successor` is ``node1``.
    
    This operation is the same as :meth:`~AbstractNode.contract_between`.

    Parameters
    ----------
    node1 : AbstractNode
        First node of the contraction. Its non-contracted edges will appear
        first in the list of inherited edges of the resultant node.
    node2 : AbstractNode
        Second node of the contraction. Its non-contracted edges will appear
        last in the list of inherited edges of the resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 7, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['right'] ^ nodeB['left']
    >>> result = tk.contract_between(nodeA, nodeB)
    >>> result.shape
    torch.Size([100, 10, 7])
    """
    return contract_edges_op(None, node1, node2)


contract_between_node = copy_func(contract_between)
contract_between_node.__doc__ = \
    """
    Contracts all edges shared between two nodes. Batch contraction is
    automatically performed when both nodes have batch edges with the same
    names. It can also be performed using the operator ``@``.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges"``.
    The node that keeps information about the :class:`Successor` is ``self``.

    Parameters
    ----------
    node2 : AbstractNode
        Second node of the contraction. Its non-contracted edges will appear
        last in the list of inherited edges of the resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 7, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['right'] ^ nodeB['left']
    >>> result = nodeA @ nodeB
    >>> result.shape
    torch.Size([100, 10, 7])
    """

AbstractNode.__matmul__ = contract_between_node
AbstractNode.contract_between = contract_between_node


def contract_between_(node1: AbstractNode,
                      node2: AbstractNode) -> Node:
    """
    In-place version of :func:`contract_between`.

    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges_ip"``.
    
    This operation is the same as :meth:`~AbstractNode.contract_between_`.

    Parameters
    ----------
    node1 : AbstractNode
        First node of the contraction. Its non-contracted edges will appear
        first in the list of inherited edges of the resultant node.
    node2 : AbstractNode
        Second node of the contraction. Its non-contracted edges will appear
        last in the list of inherited edges of the resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 7, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['right'] ^ nodeB['left']
    >>> result = tk.contract_between_(nodeA, nodeB)
    >>> result.shape
    torch.Size([100, 10, 7])
    
    ``nodeA`` and ``nodeB`` have been removed from the network.
    
    >>> nodeA.network is None
    True
    
    >>> nodeB.network is None
    True
    
    >>> del nodeA
    >>> del nodeB
    """
    result = contract_between(node1, node2)
    result.reattach_edges(override=True)
    result._unrestricted_set_tensor(result.tensor.detach())
    
    nodes = set([node1, node2])

    # Delete nodes (and their edges) from the TN
    net = result.network
    for node in nodes:
        net.delete_node(node)

    # Add edges of result to the TN
    for res_edge in result._edges:
        net._add_edge(res_edge)

    # Transform resultant to leaf nodes
    result._leaf = True
    del net._resultant_nodes[result._name]
    net._leaf_nodes[result._name] = result
    
    for node in nodes:
        node._successors = dict()
    net._seq_ops = []

    # Remove resultant name
    result.name = 'contract_edges_ip'

    return result


contract_between_node_ = copy_func(contract_between_)
contract_between_node_.__doc__ = \
    """
    In-place version of :func:`~AbstractNode.contract_between`.
    
    Following the **PyTorch** convention, names of functions ended with an
    underscore indicate **in-place** operations.
    
    Nodes ``resultant`` from this operation are called ``"contract_edges_ip"``.

    Parameters
    ----------
    node2 : AbstractNode
        Second node of the contraction. Its non-contracted edges will appear
        last in the list of inherited edges of the resultant node.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 7, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    ...
    >>> _ = nodeA['right'] ^ nodeB['left']
    >>> result = nodeA.contract_between_(nodeB)
    >>> result.shape
    torch.Size([100, 10, 7])
    
    ``nodeA`` and ``nodeB`` have been removed from the network.
    
    >>> nodeA.network is None
    True
    
    >>> nodeB.network is None
    True
    
    >>> del nodeA
    >>> del nodeB
    """

AbstractNode.contract_between_ = contract_between_node_


#####################################   STACK   ###############################
# MARK: stack
def _check_first_stack(nodes: Sequence[AbstractNode]) -> Optional[Successor]:
    if not nodes:
        raise ValueError('`nodes` should be a non-empty sequence of nodes')
    
    args = (tuple(nodes),)
    successors = nodes[0]._successors.get('stack')
    if not successors:
        return None
    return successors.get(args)


def _stack_first(nodes: Sequence[AbstractNode]) -> StackNode:
    all_leaf = True           # Check if all the nodes are leaf
    all_non_param = True      # Check if all the nodes are non-parametric
    all_param = True          # Check if all the nodes are parametric
    all_same_ref = True       # Check if all the nodes' memories are stored in
                              # the same reference node's memory
    node_ref_is_stack = True  # Check if the shared reference node is a stack
    stack_node_ref = None     # In the case above, the reference node
    stack_indices = []        # In the case above, stack indices of each node in
                              # the reference node's memory

    if not (isinstance(nodes, Sequence) and isinstance(nodes[0], AbstractNode)):
        raise TypeError('`nodes` should be a list or tuple of AbstractNodes')

    net = nodes[0]._network
    for node in nodes:
        if not node._leaf:
            all_leaf = False

        if isinstance(node, ParamNode):
            all_non_param = False
        else:
            all_param = False

        if all_same_ref:
            if node._tensor_info['address'] is None:
                node_ref = node._tensor_info['node_ref']

                if stack_node_ref is None:
                    stack_node_ref = node_ref
                else:
                    if node_ref != stack_node_ref:
                        all_same_ref = False
                        continue

                if not isinstance(node_ref, (StackNode, ParamStackNode)):
                    all_same_ref = False
                    node_ref_is_stack = False
                    continue
                
                aux_index = node._tensor_info['index']
                if isinstance(aux_index, int):
                    stack_indices.append(aux_index)
                else:
                    stack_indices.append(aux_index[0])

            else:
                all_same_ref = False

    if all_param and node_ref_is_stack and net._auto_stack:
        stack_node = ParamStackNode(nodes=nodes,
                                    name='virtual_result_stack',
                                    virtual=True)
    else:
        stack_node = StackNode._create_resultant(nodes=nodes,
                                                 name='stack')
        
    # Stack nodes' tensors
    nodes_tensors = [node.tensor for node in nodes]
    
    # Check if all dims are the same
    same_dims = True
    max_shape = list(nodes_tensors[0].shape)
    for tensor in nodes_tensors[1:]:
        for idx, dim in enumerate(tensor.shape):
            if same_dims and (dim != max_shape[idx]):
                same_dims = False
            if dim > max_shape[idx]:
                max_shape[idx] = dim
    
    # If not, pad all tensors with zeros to the maximum dims and stack
    lst_pads = []
    if not same_dims:
        for idx, tensor in enumerate(nodes_tensors):
            pad = []
            if tensor.shape != max_shape:
                for max_dim, dim in zip(max_shape, tensor.shape):
                    pad += [0, max_dim - dim]
                pad.reverse()
                lst_pads.append(pad)
                nodes_tensors[idx] = nn.functional.pad(tensor, pad)
                # NOTE: nn.functional.pad induces non-deterministic
                # behaviour in its backward pass on CUDA

    # Both conditions can only be satisfied in index_mode
    if all_same_ref:
        # Memory of stack is just a reference to the stack_node_ref
        stack_indices = list2slice(stack_indices)

        del net._memory_nodes[stack_node._tensor_info['address']]
        stack_node._tensor_info['address'] = None
        stack_node._tensor_info['node_ref'] = stack_node_ref

        index = [stack_indices]
        if stack_node_ref.shape[1:] != stack_node.shape[1:]:
            for i, (max_dim, dim) in enumerate(zip(stack_node_ref._shape[1:],
                                                   stack_node._shape[1:])):
                if stack_node._axes[i + 1]._batch:
                    # Admit any size in batch edges
                    index.append(slice(0, None))
                else:
                    index.append(slice(max_dim - dim, max_dim))
        stack_node._tensor_info['index'] = index

    else:
        if all_leaf and (all_param or all_non_param) \
                and node_ref_is_stack and net._auto_stack:
            # Stacked nodes' memories are replaced by a reference to a slice
            # of the resultant stack_node
            for i, node in enumerate(nodes):
                if node._tensor_info['address'] is not None:
                    del net._memory_nodes[node._tensor_info['address']]
                node._tensor_info['address'] = None
                node._tensor_info['node_ref'] = stack_node
                index = [i]
                for j, (max_dim, dim) in enumerate(zip(stack_node._shape[1:],
                                                       node._shape)):
                    if node._axes[j]._batch:
                        # Admit any size in batch edges
                        index.append(slice(0, None))
                    else:
                        index.append(slice(max_dim - dim, max_dim))
                node._tensor_info['index'] = index

                if all_param:
                    delattr(net, 'param_' + node._name)

        # Record in inverse_memory while tracing
        if net._tracing:
            for node in nodes:
                node._record_in_inverse_memory()

    # Create successor
    args = (tuple(nodes),)
    successor = Successor(node_ref=tuple([node.node_ref() for node in nodes]),
                          index=tuple([node._tensor_info['index'] for node in nodes]),
                          child=stack_node,
                          hints={'all_same_ref': all_same_ref,
                                 'all_leaf':
                                     all_leaf and
                                     (all_param or all_non_param) and
                                     node_ref_is_stack,
                                 'same_dims': same_dims,
                                 'lst_pads': lst_pads,
                                 'auto_stack': net._auto_stack})

    # Add successor to parent
    if 'stack' in nodes[0]._successors:
        nodes[0]._successors['stack'].update({args: successor})
    else:
        nodes[0]._successors['stack'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('stack', args))

    return stack_node


def _stack_next(successor: Successor,
                nodes: Sequence[AbstractNode]) -> StackNode:
    child = successor.child
    hints = successor.hints
    if hints['all_same_ref'] or (hints['all_leaf'] and hints['auto_stack']):
        return child
    
    if hints['same_dims']:
        nodes_tensors = list(
            starmap(lambda nr, idx, node:
                node._direct_get_tensor(nr, idx),
                zip(successor.node_ref, successor.index, nodes))
            )
    else:
        nodes_tensors = list(
            starmap(lambda nr, idx, node, pad:
                nn.functional.pad(node._direct_get_tensor(nr, idx), pad),
                zip(successor.node_ref, successor.index, nodes, hints['lst_pads']))
            )
    stack_tensor = torch.stack(nodes_tensors)

    # stack_tensor = stack_unequal_tensors([node._direct_tensor for node in nodes])
    child._direct_set_tensor(stack_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if nodes[0]._network._traced:
        for node_ref, node in zip(successor.node_ref, nodes):
            node._check_inverse_memory(node_ref)

    return child


stack_op = Operation('stack', _check_first_stack, _stack_first, _stack_next)


def stack(nodes: Sequence[AbstractNode]):
    """
    Creates a :class:`StackNode` or :class:`ParamStackNode` by stacking a
    collection of :class:`Nodes <Nodes>` or :class:`ParamNodes <ParamNode>`,
    respectively. Restrictions that are applied to the nodes in order to be
    `stackable` are the same as in :class:`StackNode`.

    The stack dimension will be the first one in the ``resultant`` node.
    
    See :class:`ParamStackNode` and :class:`TensorNetwork` to learn how the
    :meth:`~TensorNetwork.auto_stack` mode affects the computation of
    :func:`stack`.
    
    If this operation is used several times with the same input nodes, but their
    dimensions can change from one call to another, this will lead to undesired
    behaviour. The network should be :meth:`~tensorkrwoch.TensorNetwork.reset`.
    This situation should be avoided in the
    :meth:`~tensorkrowch.TensorNetwork.contract` method. Otherwise it will fail
    in subsequent calls to ``contract`` or :meth:`~tensorkrowch.TensorNetwork.forward`
    
    Nodes ``resultant`` from this operation are called ``"stack"``. If this
    operation returns a ``virtual`` :class:`ParamStackNode`, it will be called
    ``"virtual_result_stack"``. See :class:AbstractNode` to learn about this
    **reserved name**.  The node that keeps information about the
    :class:`Successor` is ``nodes[0]``, the first stacked node.

    Parameters
    ----------
    nodes : list[AbstractNode] or tuple[AbstractNode]
        Sequence of nodes that are to be stacked. They must be of the same type,
        have the same rank and axes names, be in the same tensor network, and
        have edges with the same types.
        
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodes = [tk.randn(shape=(2, 4, 2),
    ...                   axes_names=('left', 'input', 'right'),
    ...                   network=net)
    ...          for _ in range(10)]
    >>> stack_node = tk.stack(nodes)
    >>> stack_node.shape
    torch.Size([10, 2, 4, 2])
    """
    return stack_op(nodes)


##################################   UNBIND   #################################
# MARK: unbind
def _check_first_unbind(node: AbstractStackNode) -> Optional[Successor]:
    args = (node,)
    successors = node._successors.get('unbind')
    if not successors:
        return None
    return successors.get(args)


def _unbind_first(node: AbstractStackNode) -> List[Node]:
    if not isinstance(node, (StackNode, ParamStackNode)):
        raise TypeError('Cannot unbind node if it is not a (Param)StackNode')

    tensors = torch.unbind(node.tensor)
    new_nodes = []

    # Invert structure of node.edges_lists
    edges_lists = []
    node1_lists = []
    batch_ids = []
    for i, edge in enumerate(node._edges[1:]):
        if isinstance(edge, StackEdge):
            edges_lists.append(edge._edges)
            node1_lists.append(edge._node1_list)
            if edge.is_batch():
                # Save position of batch edge, whose dimension might change
                batch_ids.append(i)
        else:
            edges_lists.append([edge] * len(tensors))
            node1_lists.append([True] * len(tensors))
    
    if node._edges[1:]:
        lst = list(zip(tensors,
                       list(zip(*edges_lists)),
                       list(zip(*node1_lists))))
    else:
        lst = [(t, [], []) for t in tensors]

    net = node._network
    for i, (tensor, edges, node1_list) in enumerate(lst):
        new_node = Node._create_resultant(axes_names=node.axes_names[1:],
                                          name='unbind',
                                          network=net,
                                          tensor=tensor,
                                          edges=list(edges),
                                          node1_list=list(node1_list))
        new_nodes.append(new_node)
        
    # Check if all nodes have the same shape or have to be cropped
    same_dims = True
    for i in range(len(new_nodes[:-1])):
        if new_nodes[i].shape != new_nodes[i + 1].shape:
            same_dims = False
            break
    
    lst_crops = []
    if not same_dims:
        for i, new_node in enumerate(new_nodes):
            index = []
            for j, dim in enumerate(tensors[i].shape):
                edge = new_node.get_edge(j)

                if edge.is_batch():
                    index.append(slice(0, None))
                else:  #dim >= edge.size():
                    index.append(slice(dim - edge.size(), dim))
            lst_crops.append(index)

    if not net._auto_unbind:
        # Record in inverse_memory while tracing
        if net._tracing:
            node._record_in_inverse_memory()

    else:  # index_mode
        if node._tensor_info['address'] is None:
            node_ref = node._tensor_info['node_ref']
        else:
            node_ref = node

        for i, new_node in enumerate(new_nodes):
            if new_node._tensor_info['address'] is not None:
                del new_node._network._memory_nodes[
                    new_node._tensor_info['address']]
            new_node._tensor_info['address'] = None
            new_node._tensor_info['node_ref'] = node_ref

            if node_ref == node:
                index = [i]
                for j, (max_dim, dim) in enumerate(zip(node._shape[1:],
                                                       new_node._shape)):
                    if new_node._axes[j]._batch:
                        # Admit any size in batch edges
                        index.append(slice(0, None))
                    else:
                        index.append(slice(max_dim - dim, max_dim))

            else:
                node_index = node._tensor_info['index']
                aux_slice = node_index[0]
                if isinstance(aux_slice, list):
                    index = [aux_slice[i]]
                else:
                    index = [range(aux_slice.start,
                                   aux_slice.stop,
                                   aux_slice.step)[i]]

                if node_index[1:]:
                    # If node is indexing from the original stack
                    for j, (aux_slice, dim) in enumerate(zip(node_index[1:],
                                                             new_node._shape)):
                        if new_node._axes[j]._batch:
                            # Admit any size in batch edges
                            index.append(slice(0, None))
                        else:
                            index.append(slice(aux_slice.stop - dim,
                                               aux_slice.stop))

                else:
                    # If node has the same shape as the original stack
                    for j, (max_dim, dim) in enumerate(zip(node._shape[1:],
                                                           new_node._shape)):
                        if new_node._axes[j]._batch:
                            # Admit any size in batch edges
                            index.append(slice(0, None))
                        else:
                            index.append(slice(max_dim - dim, max_dim))

            if len(index) == 1:
                index = index[0]
            new_node._tensor_info['index'] = index

    # Create successor
    args = (node,)
    successor = Successor(node_ref=node.node_ref(),
                          index=node._tensor_info['index'],
                          child=new_nodes,
                          hints={'batch_ids': batch_ids,
                                 'same_dims': same_dims,
                                 'lst_crops': lst_crops})

    # Add successor to parent
    if 'unbind' in node._successors:
        node._successors['unbind'].update({args: successor})
    else:
        node._successors['unbind'] = {args: successor}

    # Add operation to list of performed operations of TN
    net._seq_ops.append(('unbind', args))

    # Returns copy in order not to modify the successor
    # if the returned list gets modified by any means
    return new_nodes[:]


def _unbind_next(successor: Successor, node: AbstractStackNode) -> List[Node]:
    net = node._network
    if not net._auto_unbind:
        node_tensor = node._direct_get_tensor(successor.node_ref,
                                              successor.index)
        tensors = torch.unbind(node_tensor)
        children = successor.child
        hints = successor.hints
        
        if hints['same_dims']:
            for tensor, child in zip(tensors, children):
                child._direct_set_tensor(tensor)
        else:
            for tensor, child, crop in zip(tensors, children, hints['lst_crops']):
                child._direct_set_tensor(tensor[crop])

        # Record in inverse_memory while contracting, if network is traced
        # (to delete memory if possible)
        if net._traced:
            node._check_inverse_memory(successor.node_ref)
        return children[:]

    else:  # index_mode
        children = successor.child
        batch_ids = successor.hints['batch_ids']
        diff_batches = []

        for i, j in enumerate(batch_ids):
            if children[0]._shape[j] != node._shape[j + 1]:
                batch_ids[j] = i
                diff_batches.append((i, node._shape[i + 1]))

        for child in children:
            shape = list(child._shape)
            for i, size in diff_batches:
                shape[i] = size
            child._shape = Size(shape)

        return children[:]


unbind_op = Operation('unbind',
                      _check_first_unbind,
                      _unbind_first,
                      _unbind_next)


def unbind(node: AbstractStackNode) -> List[Node]:
    """
    Unbinds a :class:`StackNode` or :class:`ParamStackNode`, where the first
    dimension is assumed to be the stack dimension.

    If :meth:`~TensorNetwork.auto_unbind` is set to ``False``, each resultant
    node will store its own tensor. Otherwise, they will have only a reference
    to the corresponding slice of the ``(Param)StackNode``.
    
    See :class:`TensorNetwork` to learn how the ``auto_unbind`` mode affects
    the computation of :func:`unbind`.
    
    Nodes ``resultant`` from this operation are called ``"unbind"``. The node
    that keeps information about the :class:`Successor` is ``node``.

    Parameters
    ----------
    node : StackNode or ParamStackNode
        Node that is to be unbound.

    Returns
    -------
    list[Node]
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodes = [tk.randn(shape=(2, 4, 2),
    ...                   axes_names=('left', 'input', 'right'),
    ...                   network=net)
    ...          for _ in range(10)]
    >>> data = [tk.randn(shape=(4,),
    ...                  axes_names=('feature',),
    ...                  network=net)
    ...         for _ in range(10)]
    ...
    >>> for i in range(10):
    ...     _ = nodes[i]['input'] ^ data[i]['feature']
    ...
    >>> stack_nodes = tk.stack(nodes)
    >>> stack_data = tk.stack(data)
    ...
    >>> # It is necessary to re-connect stacks
    >>> _ = stack_nodes['input'] ^ stack_data['feature']
    >>> result = tk.unbind(stack_nodes @ stack_data)
    >>> print(result[0].name)
    unbind_0

    >>> result[0].axes
    [Axis( left (0) ), Axis( right (1) )]

    >>> result[0].shape
    torch.Size([2, 2])
    """
    return unbind_op(node)


unbind_node = copy_func(unbind)
unbind_node.__doc__ = \
    """
    Unbinds a :class:`StackNode` or :class:`ParamStackNode`, where the first
    dimension is assumed to be the stack dimension.

    If :meth:`~TensorNetwork.auto_unbind` is set to ``False``, each resultant
    node will store its own tensor. Otherwise, they will have only a reference
    to the corresponding slice of the ``(Param)StackNode``.
    
    See :class:`TensorNetwork` to learn how the ``auto_unbind`` mode affects
    the computation of :func:`unbind`.
    
    Nodes ``resultant`` from this operation are called ``"unbind"``. The node
    that keeps information about the :class:`Successor` is ``self``.

    Returns
    -------
    list[Node]
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodes = [tk.randn(shape=(2, 4, 2),
    ...                   axes_names=('left', 'input', 'right'),
    ...                   network=net)
    ...          for _ in range(10)]
    >>> data = [tk.randn(shape=(4,),
    ...                  axes_names=('feature',),
    ...                  network=net)
    ...         for _ in range(10)]
    ...
    >>> for i in range(10):
    ...     _ = nodes[i]['input'] ^ data[i]['feature']
    ...
    >>> stack_nodes = tk.stack(nodes)
    >>> stack_data = tk.stack(data)
    ...
    >>> # It is necessary to re-connect stacks
    >>> _ = stack_nodes['input'] ^ stack_data['feature']
    >>> result = stack_nodes @ stack_data
    >>> result = result.unbind()
    >>> print(result[0].name)
    unbind_0

    >>> result[0].axes
    [Axis( left (0) ), Axis( right (1) )]

    >>> result[0].shape
    torch.Size([2, 2])
    """

StackNode.unbind = unbind_node
ParamStackNode.unbind = unbind_node


##################################   EINSUM   #################################
# MARK: einsum
def _check_first_einsum(string: Text,
                        *nodes: AbstractNode) -> Optional[Successor]:
    if not nodes:
        raise ValueError('No nodes were provided')
    
    args = (string, *nodes)
    successors = nodes[0]._successors.get('einsum')
    if not successors:
        return None
    return successors.get(args)


def _einsum_first(string: Text, *nodes: AbstractNode) -> Node:
    for i in range(len(nodes[:-1])):
        if nodes[i]._network != nodes[i + 1]._network:
            raise ValueError('All `nodes` must be in the same network')

    if '->' not in string:
        raise ValueError('Einsum `string` should have an arrow `->` separating '
                         'inputs and output strings')

    input_strings = string.split('->')[0].split(',')
    if len(input_strings) != len(nodes):
        raise ValueError('Number of einsum subscripts must be equal to the '
                         'number of operands')
    if len(string.split('->')) >= 2:
        output_string = string.split('->')[1]
    else:
        output_string = ''

    # Used for counting appearances of output subscripts in the input strings
    output_dict = dict(zip(output_string, [0] * len(output_string)))

    output_char_index = dict(zip(output_string, range(len(output_string))))

    # Used for counting how many times a contracted edge's
    # subscript appears among input strings
    contracted_edges = dict()

    # Used for counting how many times a batch edge's
    # subscript appears among input strings
    batch_edges = dict()

    axes_names = dict(zip(range(len(output_string)),
                          [None] * len(output_string)))
    edges = dict(zip(range(len(output_string)),
                     [None] * len(output_string)))
    node1_list = dict(zip(range(len(output_string)),
                          [None] * len(output_string)))

    for i, input_string in enumerate(input_strings):
        for j, char in enumerate(input_string):
            if char not in output_dict:
                edge = nodes[i][j]
                if char not in contracted_edges:
                    contracted_edges[char] = [edge]
                else:
                    if len(contracted_edges[char]) >= 2:
                        raise ValueError(f'Subscript {char} appearing more than'
                                         ' once in the input should be a batch '
                                         'index, but it does not appear among '
                                         'the output subscripts')
                    if edge != contracted_edges[char][0]:
                        if isinstance(edge, StackEdge) and \
                                isinstance(contracted_edges[char][0], StackEdge):
                            edge = edge ^ contracted_edges[char][0]
                        else:
                            raise ValueError(f'Subscript {char} appears in two '
                                             'nodes that do not share a connected'
                                             ' edge at the specified axis')
                    contracted_edges[char].append(edge)
            else:
                edge = nodes[i][j]
                if output_dict[char] == 0:
                    if edge.is_batch():
                        batch_edges[char] = 0
                    k = output_char_index[char]
                    axes_names[k] = nodes[i]._axes[j]._name
                    edges[k] = edge
                    node1_list[k] = nodes[i]._axes[j].is_node1()
                output_dict[char] += 1
                if (char in batch_edges) and edge.is_batch():
                    batch_edges[char] += 1

    for char in output_dict:
        if output_dict[char] == 0:
            raise ValueError(f'Output subscript {char} must appear among '
                             f'the input subscripts')
        if output_dict[char] > 1:
            if char in batch_edges:
                if batch_edges[char] < output_dict[char]:
                    raise ValueError(f'Subscript {char} used as batch, but some'
                                     ' of those edges are not batch edges')
            else:
                raise ValueError(f'Subscript {char} used as batch, but none '
                                 f'of those edges is a batch edge')

    for char in contracted_edges:
        if len(contracted_edges[char]) == 1:
            raise ValueError(f'Subscript {char} appears only once in the input '
                             f'but none among the output subscripts')

    input_string = ','.join(input_strings)
    einsum_string = input_string + '->' + output_string
    tensors = [node.tensor for node in nodes]
    path, _ = opt_einsum.contract_path(einsum_string, *tensors)
    new_tensor = opt_einsum.contract(einsum_string, *tensors, optimize=path)

    all_stack = True
    all_non_stack = True
    for node in nodes:
        if isinstance(node, (StackNode, ParamStackNode)):
            all_stack &= True
            all_non_stack &= False
        else:
            all_stack &= False
            all_non_stack &= True

    if not (all_stack or all_non_stack):
        raise TypeError('Cannot operate (Param)StackNode\'s with '
                        'other (non-stack) nodes')

    if all_stack:
        new_node = StackNode._create_resultant(axes_names=list(axes_names.values()),
                                               name='einsum',
                                               network=nodes[0]._network,
                                               tensor=new_tensor,
                                               edges=list(edges.values()),
                                               node1_list=list(node1_list.values()))
    else:
        new_node = Node._create_resultant(axes_names=list(axes_names.values()),
                                          name='einsum',
                                          network=nodes[0]._network,
                                          tensor=new_tensor,
                                          edges=list(edges.values()),
                                          node1_list=list(node1_list.values()))

    # Create successor
    args = (string, *nodes)
    successor = Successor(node_ref=tuple([node.node_ref() for node in nodes]),
                          index=tuple([node._tensor_info['index'] for node in nodes]),
                          child=new_node,
                          hints={'einsum_string': einsum_string,
                                 'path': path})

    # Add successor to parent
    if 'einsum' in nodes[0]._successors:
        nodes[0]._successors['einsum'].update({args: successor})
    else:
        nodes[0]._successors['einsum'] = {args: successor}

    # Add operation to list of performed operations of TN
    net = nodes[0]._network
    net._seq_ops.append(('einsum', args))

    # Record in inverse_memory while tracing
    if net._tracing:
        for node in nodes:
            node._record_in_inverse_memory()

    return new_node


def _einsum_next(successor: Successor,
                 string: Text,
                 *nodes: AbstractNode) -> Node:
    hints = successor.hints
    
    tensors = list(
        starmap(lambda nr, idx, node: node._direct_get_tensor(nr, idx),
                zip(successor.node_ref, successor.index, nodes))
    )
    new_tensor = opt_einsum.contract(hints['einsum_string'], *tensors,
                                     optimize=hints['path'])

    child = successor.child
    child._direct_set_tensor(new_tensor)

    # Record in inverse_memory while contracting, if network is traced
    # (to delete memory if possible)
    if nodes[0]._network._traced:
        for node_ref, node in zip(successor.node_ref, nodes):
            node._check_inverse_memory(node_ref)

    return child


einsum_op = Operation('einsum',
                      _check_first_einsum,
                      _einsum_first,
                      _einsum_next)


def einsum(string: Text, *nodes: Sequence[AbstractNode]) -> Node:
    r"""
    Performs einsum contraction based on `opt_einsum
    <https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.html>`_.
    This operation facilitates contracting several nodes at once, specifying
    directly the order of appearance of the resultant edges. Without this
    operation, several contractions and permutations would be needed.

    Since it adapts a tensor operation for nodes, certain nodes' properties are
    first checked. Thus, it verifies that all edges are correctly connected and
    all nodes are in the same network. It also performs batch contraction
    whenever corresponding edges are batch edges.
    
    Nodes ``resultant`` from this operation are called ``"einsum"``. The node
    that keeps information about the :class:`Successor` is ``nodes[0]``, the
    first node involved in the operation.

    Parameters
    ----------
    string : str
        Einsum-like string indicating how the contraction should be performed.
        It consists of a comma-separated list of inputs and an output separated
        by an arrow. For instance, the contraction

        .. math::

            T_{j,l} = \sum_{i,k,m}{A_{i,j,k}B_{k,l,m}C_{i,m}}

        can be expressed as::

            string = 'ijk,klm,im->jl'
    nodes : AbstractNode...
        Nodes that are involved in the contraction. Should appear in the same
        order as it is specified in the ``string``. They should either be all
        ``(Param)StackNode``'s or none of them be a ``(Param)StackNode``.

    Returns
    -------
    Node
    
    Examples
    --------
    >>> nodeA = tk.randn(shape=(10, 15, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeA')
    >>> nodeB = tk.randn(shape=(15, 7, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeB')
    >>> nodeC = tk.randn(shape=(7, 10, 100),
    ...                  axes_names=('left', 'right', 'batch'),
    ...                  name='nodeC')
    ...
    >>> _ = nodeA['right'] ^ nodeB['left']
    >>> _ = nodeB['right'] ^ nodeC['left']
    >>> _ = nodeC['right'] ^ nodeA['left']
    ...
    >>> result = tk.einsum('ijb,jkb,kib->b', nodeA, nodeB, nodeC)
    >>> result.shape
    torch.Size([100])
    """
    return einsum_op(string, *nodes)


##############################   STACKED EINSUM   #############################
def stacked_einsum(string: Text,
                   *nodes_lists: List[AbstractNode]) -> List[Node]:
    r"""
    Applies the same :func:`einsum` operation (same ``string``) to a sequence
    of groups of nodes (all groups having the same amount of nodes, with the
    same properties, etc.). That is, it stacks these groups of nodes into a
    single collection of ``StackNodes`` that is then contracted via
    :func:`einsum` (using the stack dimensions as **batch**), and
    :func:`unbound <unbind>` afterwards.

    Parameters
    ----------
    string : str
        Einsum-like string indicating how the contraction should be performed.
        It consists of a comma-separated list of inputs and an output separated
        by an arrow. For instance, the contraction

        .. math::

            T_{j,l} = \sum_{i,k,m}{A_{i,j,k}B_{k,l,m}C_{i,m}}

        can be expressed as::

            string = 'ijk,klm,im->jl'
    nodes_lists : List[Node or ParamNode]...
        Lists of nodes that are involved in the contraction. Should appear in
        the same order as it is specified in the ``string``.

    Returns
    -------
    list[Node]
    
    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> nodesA = [tk.randn(shape=(10, 15, 100),
    ...                    axes_names=('left', 'right', 'batch'),
    ...                    name='nodeA',
    ...                    network=net)
    ...           for _ in range(10)]
    >>> nodesB = [tk.randn(shape=(15, 7, 100),
    ...                    axes_names=('left', 'right', 'batch'),
    ...                    name='nodeB',
    ...                    network=net)
    ...           for _ in range(10)]
    >>> nodesC = [tk.randn(shape=(7, 10, 100),
    ...                    axes_names=('left', 'right', 'batch'),
    ...                    name='nodeC',
    ...                    network=net)
    ...           for _ in range(10)]
    ...
    >>> for i in range(10):
    ...     _ = nodesA[i]['right'] ^ nodesB[i]['left']
    ...     _ = nodesB[i]['right'] ^ nodesC[i]['left']
    ...     _ = nodesC[i]['right'] ^ nodesA[i]['left']
    ...
    >>> result = tk.stacked_einsum('ijb,jkb,kib->b', nodesA, nodesB, nodesC)
    >>> len(result)
    10
    
    >>> result[0].shape
    torch.Size([100])
    """
    stacks_list = []
    for nodes_list in nodes_lists:
        stacks_list.append(stack(nodes_list))

    input_strings = string.split('->')[0].split(',')
    output_string = string.split('->')[1]

    i = 0
    stack_char = opt_einsum.get_symbol(i)
    for input_string in input_strings:
        for input_char in input_string:
            if input_char == stack_char:
                i += 1
                stack_char = opt_einsum.get_symbol(i)

    input_strings = list(map(lambda s: stack_char + s, input_strings))
    input_string = ','.join(input_strings)
    output_string = stack_char + output_string
    string = input_string + '->' + output_string

    result = einsum(string, *stacks_list)
    unbound_result = unbind(result)
    return unbound_result
