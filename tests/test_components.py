"""
This script contains tests for components:

    * TestAxis
    * TestInitNode
    * TestInitParamNode
    * TestNodeName
    * TestSetTensorNode
    * TestSettensorparamNode
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk


class TestAxis:

    def test_same_name(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('my_axis', 'my_axis'),
                       name='my_node')
        assert node.axes_names == ['my_axis_0', 'my_axis_1']

    def test_same_name_empty(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('', ''),
                       name='my_node')
        assert node.axes_names == ['_0', '_1']

    def test_batch_axis(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('batch', 'axis'),
                       name='my_node')
        assert node.axes_names == ['batch', 'axis']
        assert node['batch'].is_batch()
        assert not node['axis'].is_batch()

    def test_stack_axis(self):
        # Cannot put name 'stack' to an axis, that is reserved
        # for edges in StackNoes
        with pytest.raises(ValueError):
            node = tk.Node(shape=(3, 3),
                           axes_names=('stack', 'axis'),
                           name='my_node',
                           init_method='randn')

        node = tk.Node(shape=(3,),
                       axes_names=('axis',),
                       name='my_node',
                       init_method='randn')
        stack = tk.stack([node])

        assert stack.axes_names == ['stack', 'axis']
        assert stack['stack'].is_batch()
        assert not stack['axis'].is_batch()
        
        with pytest.raises(ValueError):
            stack.get_axis('axis').name = 'other_stack'
            
        # Name of axis "stack" cannot be changed
        with pytest.raises(ValueError):
            stack.get_axis('stack').name = 'axis'

    def test_change_axis_name(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('axis', 'axis1'),
                       name='my_node')
        assert node.axes_names == ['axis', 'axis1']

        node.get_axis('axis1').name = 'axis2'
        assert node.axes_names == ['axis', 'axis2']

        node.get_axis('axis2').name = 'axis'
        assert node.axes_names == ['axis_0', 'axis_1']

    def test_change_name_batch(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('batch', 'axis'),
                       name='my_node')
        assert node.axes_names == ['batch', 'axis']
        assert node.get_axis('batch').is_batch()
        assert not node.get_axis('axis').is_batch()

        # Batch attribute depends on the name,
        # only axis with word "batch" or "stack"
        # in the name are batch axis
        node.get_axis('batch').name = 'new_axis'
        assert node.axes_names == ['new_axis', 'axis']
        assert not node.get_axis('new_axis').is_batch()
        assert not node.get_axis('axis').is_batch()

        node.get_axis('axis').name = 'new_batch'
        assert node.axes_names == ['new_axis', 'new_batch']
        assert not node.get_axis('new_axis').is_batch()
        assert node.get_axis('new_batch').is_batch()


class TestInitNode:

    def test_init_node_empty(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='my_node')

        assert node.name == 'my_node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None

        assert node.tensor is None
        assert node._tensor_info == {'address': 'my_node',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']

        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()

    def test_init_node_data(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       network=tk.TensorNetwork('my_net'),
                       data=True)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None

        assert node.tensor is None
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'my_net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 1
        assert list(net._memory_nodes.keys()) == ['node']

        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert not node.is_leaf()
        assert node.is_data()
        assert node.successors == dict()

    def test_init_node_virtual(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       network=tk.TensorNetwork('my_net'),
                       virtual=True)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None

        assert node.tensor is None
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'my_net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.virtual_nodes) == 1
        assert list(net._memory_nodes.keys()) == ['node']

        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert not node.is_leaf()
        assert node.is_virtual()
        assert node.successors == dict()

    def test_init_node_no_axis_names(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.Node(tensor=tensor)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['axis_0', 'axis_1', 'axis_2']
        assert node.rank == 3
        assert node.dtype is torch.float32

        assert torch.equal(node.tensor, tensor)
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']

        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()

    def test_init_node_errors(self):
        # A Node cannot be data and virtual at the same time
        with pytest.raises(ValueError):
            node = tk.Node(shape=(2, 5, 2),
                           data=True,
                           virtual=True)

        # Shape and tensor.shape must be equal if both are provided
        with pytest.raises(ValueError):
            node = tk.Node(shape=(2, 5, 3),
                           tensor=torch.randn(2, 5, 2))


class TestInitParamNode:

    def test_init_paramnode_empty(self):
        node = tk.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name='my_node')

        assert node.name == 'my_node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None

        assert node.tensor is None
        assert node._tensor_info == {'address': 'my_node',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']

        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()

    def test_init_paramnode_virtual(self):
        # ParamNodes can be virtual, to store memory of ParamNodes
        # (e.g. in ParamStackNodes or in Uniform TN)
        node = tk.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            network=tk.TensorNetwork('my_net'),
                            virtual=True)

    def test_init_paramnode_no_axis_names(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.ParamNode(tensor=tensor)

        assert node.name == 'paramnode'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['axis_0', 'axis_1', 'axis_2']
        assert node.rank == 3
        assert node.dtype is torch.float32

        assert torch.equal(node.tensor, nn.Parameter(tensor))
        assert node._tensor_info == {'address': 'paramnode',
                                     'node_ref': None,
                                     'full': True,
                                     'index': None}

        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']

        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)

        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()

    def test_init_paramnode_errors(self):
        # ParamNodes are always leaf nodes (or virtual)
        with pytest.raises(NotImplementedError):
            node = tk.ParamNode._create_resultant(shape=(2, 5, 2))

        # Shape and tensor.shape must be equal if both are provided
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 3),
                                tensor=torch.randn(2, 5, 2))


class TestNodeName:

    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        network=net)

        node1['right'] ^ node2['left']
        net.set_data_nodes([node1['input'], node2['input']], 1)

        return net

    def test_change_node_name(self, setup):
        net = setup

        node1 = net['node1']
        node2 = net['node2']
        assert node1.name == 'node1'
        assert node2.name == 'node2'

        node1.name = 'new_node'
        assert node1.name == 'new_node'
        assert node2.name == 'node2'

        node1.name = 'node2'
        assert node1.name == 'node2_1'
        assert node2.name == 'node2_0'

        node1.name = 'node1'
        assert node1.name == 'node1'
        assert node2.name == 'node2'

        node2.name = 'node1'
        assert node1.name == 'node1_0'
        assert node2.name == 'node1_1'

        node1.name = 'node'
        assert node1.name == 'node'
        assert node2.name == 'node1'

    def test_change_name_to_data(self, setup):
        net = setup

        node1 = net['node1']
        node2 = net['node2']
        assert node1.name == 'node1'
        assert node2.name == 'node2'

        node1.name = 'data'
        assert node1.name == 'data_2'
        assert node2.name == 'node2'

        # What determines whether a node is a data node or
        # not is the attribute `data`, not the name
        data_node = net['data_0']
        assert not node1.is_data()
        assert data_node.is_data()

    def test_change_data_name(self, setup):
        net = setup

        data_node = net['data_0']
        assert data_node.name == 'data_0'

        data_node.name = 'node'
        assert data_node.name == 'node'
        assert data_node.is_data()

    def test_use_resultant_name(self, setup):
        net = setup

        node1 = net['node1']
        node1.set_tensor()

        node2 = net['node2']
        node2.set_tensor()

        # Names of operations cannot be used as names of leaf nodes
        with pytest.raises(ValueError):
            node1.name = 'contract_edges'

    def test_name_resultant(self, setup):
        net = setup

        node1 = net['node1']
        node1.set_tensor()

        node2 = net['node2']
        node2.set_tensor()

        node3 = node1 @ node2
        assert node3.name == 'contract_edges'

        # We can change the name of resultant nodes also
        node3.name = 'node3'
        assert node3.name == 'node3'


class TestSetTensorNode:

    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'batch', 'right'),
                        name='node1',
                        network=net)

        tensor = torch.randn(2, 5, 2)
        node2 = tk.Node(axes_names=('left', 'batch', 'right'),
                        name='node2',
                        tensor=tensor,
                        network=net)
        return node1, node2, tensor

    def test_set_tensor(self, setup):
        node1, node2, tensor = setup

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.tensor = tensor
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'
        assert node2.tensor_address() == 'node2'

    def test_set_tensor_zeros(self, setup):
        node1, node2, tensor = setup

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)

        node1.set_tensor()
        assert torch.equal(node1.tensor, torch.zeros(node1.shape))
        assert node1.shape == (2, 5, 2)

    def test_change_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Changing tensor changes node1's and node2's tensor
        node1.tensor = tensor
        tensor[0, 0, 0] = 1000
        assert node1.tensor[0, 0, 0] == 1000
        assert node2.tensor[0, 0, 0] == 1000

    def test_unset_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Using unset_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.unset_tensor()
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)

        # Using tensor.setter
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.tensor = None
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)

        # Using set_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.set_tensor(None)
        assert torch.equal(node1.tensor, torch.zeros(node1.shape))
        assert node1.shape == (2, 5, 2)

    def test_set_diff_shape(self, setup):
        node1, node2, tensor = setup
        node1['left'] ^ node2['left']

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 5
        assert node1['right'].size() == 2

        # If edges are batch edges, we can set a
        # tensor with different size in those axes
        diff_tensor = torch.randn(2, 10, 2)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 10, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 10
        assert node1['right'].size() == 2

        # If edges are non-dangling, we can set a
        # tensor with different size but it is cropped
        # to match the size in the connected edges
        diff_tensor = torch.randn(5, 20, 5)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 20, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 20
        assert node1['right'].size() == 2

    def test_set_diff_shape_unrestricted(self, setup):
        node1, node2, tensor = setup
        node1['left'] ^ node2['left']

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 5
        assert node1['right'].size() == 2

        # If using _unrestricted_set_tensor, used for
        # setting tensors in resultant nodes, size of
        # edges is not updated, so that we don't change
        # the original shapes
        diff_tensor = torch.randn(2, 10, 2)
        node1._unrestricted_set_tensor(diff_tensor)
        assert node1.shape == (2, 10, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 10
        assert node1['right'].size() == 2

    def test_set_init_method(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Initialize tensor of node1
        node1.set_tensor(init_method='randn', mean=1., std=2.)
        assert node1.tensor is not None

        # Set node1's tensor as node2's tensor
        node2.tensor = node1.tensor
        assert torch.equal(node1.tensor, node2.tensor)

        # Changing node1's tensor changes node2's tensor
        node1.tensor[0, 0, 0] = 1000
        assert node2.tensor[0, 0, 0] == 1000
    
    def test_set_tensor_from(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        node2.tensor = torch.randn(node2.shape)
        assert torch.equal(node1.tensor, node2.tensor)
    
    def test_set_tensor_from_empty(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node2.set_tensor_from(node1)
        assert node1.tensor is None
        assert node2.tensor is None
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node1'  # But empty
        
        node1.tensor = torch.randn(node1.shape)
        assert torch.equal(node1.tensor, node2.tensor)
    
    def test_set_tensor_from_other_type(self, setup):
        node1, node2, tensor = setup
        
        node1 = node1.parameterize()
        with pytest.raises(TypeError):
            # Node and ParamNode cannot share tensor
            node1.set_tensor_from(node2)
            
    def test_set_tensor_from_other_network(self, setup):
        node1, node2, tensor = setup
        
        node1.network = tk.TensorNetwork()
        with pytest.raises(ValueError):
            # Cannot share tensor if they are in different networks
            node1.set_tensor_from(node2)
    
    def test_set_node_with_reference(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        # Now node1 just has a reference to node2's tensor
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        with pytest.raises(ValueError):
            # Cannot set tensor in node1 if it has a reference to node2
            node1.tensor = torch.zeros(node1.shape)
            
    def test_set_node_with_reference_reset(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        # Now node1 just has a reference to node2's tensor
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        node1.reset_tensor_address()
        assert node1.tensor_address() == 'node1'
        assert node2.tensor_address() == 'node2'
        assert torch.equal(node1.tensor, node2.tensor)
        
        node1.tensor = torch.zeros(node1.shape)
        assert not torch.equal(node1.tensor, node2.tensor)


class TestSetTensorParamNode:

    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'batch', 'right'),
                             name='node1',
                             network=net)

        tensor = torch.randn(2, 5, 2)
        node2 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'batch', 'right'),
                             name='node2',
                             tensor=tensor,
                             network=net)
        return node1, node2, tensor

    def test_set_tensor(self, setup):
        node1, node2, tensor = setup

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.tensor = tensor
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'
        assert node2.tensor_address() == 'node2'

        assert isinstance(node1.tensor, nn.Parameter)
        assert isinstance(node2.tensor, nn.Parameter)

    def test_change_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Changing tensor changes node1's and node2's tensor
        node1.tensor = tensor
        tensor[0, 0, 0] = 1000
        assert node1.tensor[0, 0, 0] == 1000
        assert node2.tensor[0, 0, 0] == 1000

    def test_unset_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Using unset_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.unset_tensor()
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)

        # Using tensor.setter
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.tensor = None
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)

        # Using set_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)

        node1.set_tensor(None)
        assert torch.equal(node1.tensor, torch.zeros(node1.shape))
        assert node1.shape == (2, 5, 2)

    def test_set_diff_shape(self, setup):
        node1, node2, tensor = setup
        node1['left'] ^ node2['left']

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 5
        assert node1['right'].size() == 2

        # If edges are batch edges, we can set a
        # tensor with different size in those axes
        diff_tensor = torch.randn(2, 10, 2)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 10, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 10
        assert node1['right'].size() == 2

        # If edges are non-dangling, we can set a
        # tensor with different size but it is cropped
        # to match the size in the connected edges
        diff_tensor = torch.randn(5, 20, 5)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 20, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 20
        assert node1['right'].size() == 2

    def test_set_diff_shape_unrestricted(self, setup):
        node1, node2, tensor = setup
        node1['left'] ^ node2['left']

        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 5
        assert node1['right'].size() == 2

        # If using _unrestricted_set_tensor, used for
        # setting tensors in resultant nodes, size of
        # edges is not updated, so that we don't change
        # the original shapes
        diff_tensor = torch.randn(2, 10, 2)
        node1._unrestricted_set_tensor(diff_tensor)
        assert node1.shape == (2, 10, 2)
        assert node1['left'].size() == 2
        assert node1['batch'].size() == 10
        assert node1['right'].size() == 2

    def test_set_init_method(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Initialize tensor of node1
        node1.set_tensor(init_method='randn', mean=1., std=2.)
        assert node1.tensor is not None

        # Set node1's tensor as node2's tensor
        node2.tensor = node1.tensor
        assert torch.equal(node1.tensor, node2.tensor)

        # Cannot change element of Parameter
        with pytest.raises(RuntimeError):
            node1.tensor[0, 0, 0] = 1000

    def test_set_parametric(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None

        # Create parametric node
        node3 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                             tensor=tensor)
        assert isinstance(node3.tensor, nn.Parameter)
        assert torch.equal(node3.tensor.data, tensor)

        # Creating parameter from tensor does not affect tensor's grad
        param = nn.Parameter(tensor)
        param.mean().backward()
        assert node3.grad is None

        # Set nn.Parameter as node3's tensor
        node3.tensor = param
        assert node3.grad is not None
        assert torch.equal(node3.grad, node3.tensor.grad)

        node3.tensor = None
        assert node3.tensor is None
        assert node3.grad is None
        
    def test_set_tensor_from(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        node2.tensor = torch.randn(node2.shape)
        assert torch.equal(node1.tensor, node2.tensor)
    
    def test_set_tensor_from_empty(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node2.set_tensor_from(node1)
        assert node1.tensor is None
        assert node2.tensor is None
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node1'  # But empty
        
        node1.tensor = torch.randn(node1.shape)
        assert torch.equal(node1.tensor, node2.tensor) 
    
    def test_set_tensor_from_other_type(self, setup):
        node1, node2, tensor = setup
        
        node1 = node1.parameterize(False)
        with pytest.raises(TypeError):
            # Node and ParamNode cannot share tensor
            node1.set_tensor_from(node2)
            
    def test_set_tensor_from_other_network(self, setup):
        node1, node2, tensor = setup
        
        node1.network = tk.TensorNetwork()
        with pytest.raises(ValueError):
            # Cannot share tensor if they are in different networks
            node1.set_tensor_from(node2)
    
    def test_set_node_with_reference(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        # Now node1 just has a reference to node2's tensor
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        with pytest.raises(ValueError):
            # Cannot set tensor in node1 if it has a reference to node2
            node1.tensor = torch.zeros(node1.shape)
            
    def test_set_node_with_reference_reset(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        assert node1.tensor_address() == 'node1'  # But empty
        assert node2.tensor_address() == 'node2'

        node1.set_tensor_from(node2)
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
        # Now node1 just has a reference to node2's tensor
        assert node1.tensor_address() == 'node2'
        assert node2.tensor_address() == 'node2'
        
        node1.reset_tensor_address()
        assert node1.tensor_address() == 'node1'
        assert node2.tensor_address() == 'node2'
        assert torch.equal(node1.tensor, node2.tensor)
        
        node1.tensor = torch.zeros(node1.shape)
        assert not torch.equal(node1.tensor, node2.tensor)


class TestMoveToNetwork:
    
    def test_change_network(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))

        net1 = node1.network
        net2 = node2.network
        assert net1 != net2
        assert len(net1.nodes) == 1
        assert len(net1.edges) == 3
        assert len(net2.nodes) == 1
        assert len(net2.edges) == 3

        node2.network = net1
        assert node1.network == node2.network
        assert len(net1.nodes) == 2
        assert len(net1.edges) == 6
        assert len(net2.nodes) == 0
        assert len(net2.edges) == 0

    def test_move_to_network(self):
        net1 = tk.TensorNetwork(name='net1')
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2),
                        network=net1)
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2),
                        network=net1)

        net2 = tk.TensorNetwork(name='net2')
        node2.network = net2
        assert node1.network == net1
        assert node2.network == net2

        node2.network = net1
        assert node1.network == net1
        assert node2.network == net1

        node1['right'] ^ node2['left']
        node2.network = net2
        assert node1.network == net2
        assert node2.network == net2
        assert len(net2.nodes) == 2
        assert len(net2.edges) == 4
        assert len(net1.nodes) == 0
        assert len(net1.edges) == 0
    
    def test_move_all(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 3),
                        axes_names=('left', 'right'),
                        name='node1',
                        network=net)
        node2 = tk.Node(shape=(3, 4),
                        axes_names=('left', 'right'),
                        name='node2',
                        network=net)
        node1['right'] ^ node2['left']
        
        assert node1.network == net
        assert node2.network == net
        
        other_net = tk.TensorNetwork()
        node1.network = other_net
        
        assert node1.network == other_net
        assert node2.network == other_net
        
    def test_move_some_leave_other(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 3),
                        axes_names=('left', 'right'),
                        name='node1',
                        network=net)
        node2 = tk.Node(shape=(3, 4),
                        axes_names=('left', 'right'),
                        name='node2',
                        network=net)
        node3 = tk.Node(shape=(4, 5),
                        axes_names=('left', 'right'),
                        name='node3',
                        network=net)
        node1['right'] ^ node2['left']
        
        assert node1.network == net
        assert node2.network == net
        assert node3.network == net
        
        other_net = tk.TensorNetwork()
        node1.network = other_net
        
        assert node1.network == other_net
        assert node2.network == other_net
        assert node3.network == net
        
    def test_move_some_leave_other_share_tensor(self):
        net = tk.TensorNetwork()
        node1 = tk.empty(shape=(2, 3),
                         axes_names=('left', 'right'),
                         name='node1',
                         network=net)
        node2 = tk.randn(shape=(3, 4),
                         axes_names=('left', 'right'),
                         name='node2',
                         network=net)
        node1.set_tensor_from(node2)
        
        assert node1.network == net
        assert node2.network == net
        
        other_net = tk.TensorNetwork()
        node1.network = other_net
        
        assert node1.network == other_net
        assert node2.network == net
        
        # When node is moved to another network, its _tensor_info is set
        # for the first time in that network. This automatically sets the
        # address as the name of the node, so the node recovers the ownership
        # of its tensor
        assert node1.tensor_address() == 'node1'
        assert node2.tensor_address() == 'node2'
        assert torch.equal(node1.tensor, node2.tensor)      


class TestMeasures:
    
    def test_sum(self):
        tensor = torch.randn(2, 3)
        node = tk.Node(axes_names=('left', 'right'),
                       tensor=tensor)
        
        assert node.sum() == tensor.sum()
        assert torch.equal(node.sum(0), tensor.sum(0))
        assert torch.equal(node.sum(1), tensor.sum(1))
        assert torch.equal(node.sum('left'), tensor.sum(0))
        assert torch.equal(node.sum('right'), tensor.sum(1))
        assert torch.equal(node.sum(['left', 'right']), tensor.sum([0, 1]))
        
    def test_mean(self):
        tensor = torch.randn(2, 3)
        node = tk.Node(axes_names=('left', 'right'),
                       tensor=tensor)
        
        assert node.mean() == tensor.mean()
        assert torch.equal(node.mean(0), tensor.mean(0))
        assert torch.equal(node.mean(1), tensor.mean(1))
        assert torch.equal(node.mean('left'), tensor.mean(0))
        assert torch.equal(node.mean('right'), tensor.mean(1))
        assert torch.equal(node.mean(['left', 'right']), tensor.mean([0, 1]))
        
    def test_std(self):
        tensor = torch.randn(2, 3)
        node = tk.Node(axes_names=('left', 'right'),
                       tensor=tensor)
        
        assert node.std() == tensor.std()
        assert torch.equal(node.std(0), tensor.std(0))
        assert torch.equal(node.std(1), tensor.std(1))
        assert torch.equal(node.std('left'), tensor.std(0))
        assert torch.equal(node.std('right'), tensor.std(1))
        assert torch.equal(node.std(['left', 'right']), tensor.std([0, 1]))
        
    def test_norm(self):
        tensor = torch.randn(2, 3)
        node = tk.Node(axes_names=('left', 'right'),
                       tensor=tensor)
        
        assert node.norm() == tensor.norm()
        assert torch.equal(node.norm(2, 0), tensor.norm(2, 0))
        assert torch.equal(node.norm(2, 1), tensor.norm(2, 1))
        assert torch.equal(node.norm(2, 'left'), tensor.norm(2, 0))
        assert torch.equal(node.norm(2, 'right'), tensor.norm(2, 1))
        assert torch.equal(node.norm(2, ['left', 'right']), tensor.norm(2, [0, 1]))


class TestConnect:

    def test_connect_edges(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')

        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network

        assert isinstance(node1[2], tk.Edge)
        assert isinstance(node2[0], tk.Edge)

        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.Edge)

        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network

        # Using connect method
        new_edge = node1[0].connect(node2[2])
        assert node1[0] == node2[2]

        new_edge.disconnect()
        assert node1[0] != node2[2]

    def test_connect_same_network(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)
        assert node1.name == 'node'

        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)

        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network

        assert isinstance(node1[2], tk.Edge)
        assert isinstance(node2[0], tk.Edge)

        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.Edge)

        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network

    def test_connect_different_sizes(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2')

        node2[0].change_size(4)
        assert node2[0].size() == 4

        with pytest.raises(ValueError):
            # Edges with different sizes cannot be connected
            node1[2] ^ node2[0]

    def test_connect_with_result(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')
        node1[2] ^ node2[0]
        node3 = node1 @ node2

        node4 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node4',
                        init_method='randn')

        # When we connect an edge from a node that is the result of an operation,
        # be aware that its edges are inherited from leaf nodes, so the connection
        # takes place between the leaf nodes.
        edge = node3[3]
        node3[3] ^ node4[0]
        assert node3[3] == edge
        assert node2[2] == node4[0]
        with pytest.raises(ValueError):
            # Raises error because nodes are not connected,
            # node3 still has the disconnected edge
            node3 @ node4

        # You can do this to overcome this issue, but is not recommended
        # practice. You should better make all the connections before starting
        # with operations
        node4[0] | node4[0]
        node3.reattach_edges(override=True)
        edge = node3[3]
        node3[3] ^ node4[0]
        assert node3[3] != edge
        assert node3[3] == node4[0]

    def test_disconnect(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')

        assert node1['right'].is_dangling()
        assert node2['left'].is_dangling()
        assert node1['right'] != node2['left']

        edge = node1['right'] ^ node2['left']
        assert not node1['right'].is_dangling()
        assert not node2['left'].is_dangling()
        assert node1['right'] == node2['left']

        edge | edge
        assert node1['right'].is_dangling()
        assert node2['left'].is_dangling()
        assert node1['right'] != node2['left']

    def test_connect_disconnect_stackedge(self):
        net = tk.TensorNetwork()
        left_nodes = []
        right_nodes = []
        for _ in range(10):
            node = tk.Node(shape=(5,),
                           axes_names=('left',),
                           init_method='randn',
                           network=net)
            left_nodes.append(node)

            node = tk.Node(shape=(5,),
                           axes_names=('right',),
                           init_method='randn',
                           network=net)
            right_nodes.append(node)

        for i in range(10):
            left_nodes[i][0] ^ right_nodes[i][0]

        left_stack = tk.stack(left_nodes)
        right_stack = tk.stack(right_nodes)

        assert isinstance(left_stack[1], tk.StackEdge)
        assert left_stack[1].is_dangling()
        assert isinstance(right_stack[1], tk.StackEdge)
        assert right_stack[1].is_dangling()

        left_stack[1] ^ right_stack[1]
        assert left_stack[1] == right_stack[1]
        assert isinstance(left_stack[1], tk.StackEdge)

        left_stack[1] | left_stack[1]
        assert left_stack[1] != right_stack[1]
        assert isinstance(left_stack[1], tk.StackEdge)
        assert left_stack[1].is_dangling()
        assert isinstance(right_stack[1], tk.StackEdge)
        assert right_stack[1].is_dangling()

    def test_connect_disconnected_stacks(self):
        net = tk.TensorNetwork()
        left_nodes = []
        right_nodes = []
        for _ in range(10):
            node = tk.Node(shape=(5,),
                           axes_names=('left',),
                           init_method='randn',
                           network=net)
            left_nodes.append(node)

            node = tk.Node(shape=(5,),
                           axes_names=('right',),
                           init_method='randn',
                           network=net)
            right_nodes.append(node)

        left_stack = tk.stack(left_nodes)
        right_stack = tk.stack(right_nodes)

        with pytest.raises(ValueError):
            left_stack[1] ^ right_stack[1]

    def test_reattach_copy_disconnect(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             network=net)
        node2 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node2',
                             init_method='randn',
                             network=net)
        node3 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node3',
                             init_method='randn',
                             network=net)
        node1['right'] ^ node2['left']
        node2['right'] ^ node3['left']

        node4 = node1 @ node2

        assert node4['right'].is_attached_to(node2)
        assert node4['right'].is_attached_to(node3)

        # Reattach a copy of the edges
        node4.reattach_edges()

        assert node4['right'].is_attached_to(node4)
        assert node4['right'].is_attached_to(node3)

        # Original edge (still in node3) is still attached to node2
        assert node4['right'] != node3['left']
        assert node3['left'].is_attached_to(node2)
        assert node3['left'].is_attached_to(node3)

        # If we disconnect node4 from node3, node3 still has its edge,
        # but node4 gets a new dangling edge
        node4['right'].disconnect()
        assert node4['right'].is_dangling()
        assert node4['right'].is_attached_to(node4)

        assert node3['left'].is_attached_to(node2)
        assert node3['left'].is_attached_to(node3)

    def test_reattach_original_disconnect(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             network=net)
        node2 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node2',
                             init_method='randn',
                             network=net)
        node3 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node3',
                             init_method='randn',
                             network=net)
        node1['right'] ^ node2['left']
        node2['right'] ^ node3['left']

        node4 = node1 @ node2

        assert node4['right'].is_attached_to(node2)
        assert node4['right'].is_attached_to(node3)

        # Reattach a copy of the edges
        node4.reattach_edges(True)

        assert node4['right'].is_attached_to(node4)
        assert node4['right'].is_attached_to(node3)

        # All edges connected to node4 are alse changed in its neighbours
        assert node4['right'] == node3['left']
        assert node3['left'].is_attached_to(node4)
        assert node3['left'].is_attached_to(node3)

        # If we disconnect node4 from node3, now both nodes get new
        # dangling edges
        node4['right'].disconnect()
        assert node4['right'].is_dangling()
        assert node4['right'].is_attached_to(node4)

        assert node3['left'].is_dangling()
        assert node3['left'].is_attached_to(node3)


class TestChangeSizeEdge:

    def test_change_size_dangling(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node',
                       init_method='randn')

        for i, edge in enumerate(node.edges):
            assert isinstance(edge, tk.Edge)
            assert edge.size() == node.shape[i]

        edge = node[0]
        edge.change_size(size=4)
        assert edge.size() == 4
        assert edge.node1.size() == (4, 5, 2)

        edge.change_size(size=2)
        assert edge.size() == 2
        assert edge.node1.size() == (2, 5, 2)

    def test_increase_size_connected(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')
        node1['right'] ^ node2['left']

        old_tensors = [node1.tensor, node2.tensor]

        edge = node2['left']
        edge.change_size(size=4)
        assert edge.size() == 4
        assert edge.node1.size() == (2, 5, 4)
        assert edge.node2.size() == (4, 5, 2)

        edge.change_size(size=2)
        assert edge.size() == 2
        assert edge.node1.size() == (2, 5, 2)
        assert edge.node2.size() == (2, 5, 2)

        new_tensors = [node1.tensor, node2.tensor]
        assert torch.allclose(old_tensors[0], new_tensors[0])
        assert torch.allclose(old_tensors[1], new_tensors[1])

    def test_decrease_size_connected(self):
        node1 = tk.Node(shape=(3, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(3, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')
        node1['right'] ^ node2['left']

        old_tensors = [node1.tensor, node2.tensor]

        edge = node2['left']
        edge.change_size(size=2)
        assert edge.size() == 2
        assert edge.node1.size() == (3, 5, 2)
        assert edge.node2.size() == (2, 5, 3)

        edge.change_size(size=3)
        assert edge.size() == 3
        assert edge.node1.size() == (3, 5, 3)
        assert edge.node2.size() == (3, 5, 3)

        new_tensors = [node1.tensor, node2.tensor]
        assert not torch.allclose(old_tensors[0], new_tensors[0])
        assert not torch.allclose(old_tensors[1], new_tensors[1])


class TestParameterize:

    def test_parameterize_node(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']

        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert node1['right'] == node2['left']

        paramnode1 = node1.parameterize()
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert net.edges == [node2['input'], node2['right'],
                             paramnode1['left'], paramnode1['input']]

        # Now `node2` and `paramnode1` share same edges
        assert paramnode1['right'] == node2['left']

        # `node1` still exists and has edges pointing to `node2`,
        # but `node2` cannot "see" it
        assert node1['right'] != node2['left']
        assert node1['right']._nodes[node1.is_node1('right')] == node2
        del node1

        assert isinstance(paramnode1, tk.ParamNode)
        assert paramnode1['left'].node1 == paramnode1
        assert isinstance(paramnode1['left'], tk.Edge)

    def test_deparameterize_paramnode(self):
        paramnode1 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                                  name='paramnode1',
                                  tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        paramnode1['right'] ^ node2['left']

        net = paramnode1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert paramnode1['right'] == node2['left']

        node1 = paramnode1.parameterize(False)
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert net.edges == [node2['input'], node2['right'],
                             node1['left'], node1['input']]

        # Now `node2` and `node1` share same edges
        assert node1['right'] == node2['left']

        # `paramnode1` still exists and has edges pointing to `node2`,
        # but `node2` cannot "see" it
        assert paramnode1['right'] != node2['left']
        assert paramnode1['right']._nodes[paramnode1.is_node1(
            'right')] == node2
        del paramnode1

        assert isinstance(node1, tk.Node)
        assert node1['left'].node1 == node1
        assert isinstance(node1['left'], tk.Edge)

    def test_parameterize_override_name(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']

        assert node1.name == 'node_0'
        assert node2.name == 'node_1'

        # Even though we are overriding a node with the same name,
        # the enumeration can change
        paramnode1 = node1.parameterize()
        assert paramnode1.name == 'node_1'
        assert node2.name == 'node_0'

    def test_parameterize_resultant(self):
        node1 = tk.randn(shape=(2, 3, 2))
        node2 = tk.randn(shape=(2, 3, 2))

        node1[2] ^ node2[0]
        node3 = node1 @ node2

        node3.name = 'paramnode3'
        paramnode3 = node3.parameterize()

        assert paramnode3.name == 'paramnode3'
        assert isinstance(paramnode3, tk.ParamNode)
        assert paramnode3[0] != node1[0]
        assert paramnode3[1] != node1[1]
        assert paramnode3[1] != node2[1]
        assert paramnode3[2] != node2[2]


class TestCopy:

    def test_copy_edge(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        edge = node1['right'] ^ node2['left']

        copy_edge = edge.copy()
        assert isinstance(edge, tk.Edge)
        assert isinstance(copy_edge, tk.Edge)
        assert copy_edge._nodes == edge._nodes
        assert copy_edge._axes == edge._axes
        assert copy_edge != edge

    def test_copy_node_empty(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2')
        node1['right'] ^ node2['left']

        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4

        copy = node1.copy()
        assert copy.tensor == node1.tensor
        assert len(net.nodes) == 3
        assert len(net.edges) == 6

        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]

        assert node2['left'].node1 == node1

    def test_copy_node(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']

        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4

        copy = node1.copy()
        assert torch.equal(copy.tensor, node1.tensor)
        assert len(net.nodes) == 3
        assert len(net.edges) == 6

        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]

        assert node2['left'].node1 == node1

    def test_copy_paramnode_empty(self):
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2')
        node1['right'] ^ node2['left']

        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4

        copy = node1.copy()
        assert copy.tensor == node1.tensor
        assert len(net.nodes) == 3
        assert len(net.edges) == 6

        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]

        assert node2['left'].node1 == node1

    def test_copy_paramnode(self):
        node1 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                             name='node1',
                             tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']

        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4

        copy = node1.copy()
        assert torch.equal(copy.tensor, node1.tensor)
        assert len(net.nodes) == 3
        assert len(net.edges) == 6

        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]

        assert node2['left'].node1 == node1
        
    def test_copy_node_preserve_name(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'

        copy = node1.copy()
        assert node1.name in copy.name
        assert copy.name == 'node_0_copy'


class TestStack:

    def test_stack_nodes_in_stacknode(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)

        stack = tk.StackNode(nodes)

        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'input', 'output']
        assert isinstance(stack['stack'], tk.Edge)
        assert isinstance(stack['input'], tk.StackEdge)
        assert isinstance(stack['output'], tk.StackEdge)
        
        for node in nodes:
            assert node.tensor_address() == node.name

    def test_stack_paramnodes_in_stacknode(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 2),
                                axes_names=('input', 'output'),
                                name='node',
                                network=net,
                                init_method='randn')
            nodes.append(node)

        stack = tk.StackNode(nodes)

        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'input', 'output']
        assert isinstance(stack['stack'], tk.Edge)
        assert isinstance(stack['input'], tk.StackEdge)
        assert isinstance(stack['output'], tk.StackEdge)
        
        for node in nodes:
            assert node.tensor_address() == node.name

    def test_stack_nodes_in_paramstacknode(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)

        stack = tk.ParamStackNode(nodes)

        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'input', 'output']
        assert isinstance(stack['stack'], tk.Edge)
        assert isinstance(stack['input'], tk.StackEdge)
        assert isinstance(stack['output'], tk.StackEdge)
        
        for node in nodes:
            assert node.tensor_address() == node.name

    def test_stack_paramnodes_in_paramstacknode(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 2),
                                axes_names=('input', 'output'),
                                name='node',
                                network=net,
                                init_method='randn')
            nodes.append(node)

        stack = tk.ParamStackNode(nodes)

        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'input', 'output']
        assert isinstance(stack['stack'], tk.Edge)
        assert isinstance(stack['input'], tk.StackEdge)
        assert isinstance(stack['output'], tk.StackEdge)
        
        # Nodes only modify their memory address if stacked
        # via operation `stack`
        for node in nodes:
            assert node.tensor_address() == node.name

    def test_error_stack_node(self):
        node = tk.Node(shape=(3, 2),
                       axes_names=('input', 'output'),
                       init_method='randn')

        # Be careful! You have to pass a list or tuple of nodes as input
        with pytest.raises(TypeError):
            stack = tk.StackNode(node)

        with pytest.raises(TypeError):
            stack = tk.ParamStackNode(node)

        stack = tk.StackNode([node])
        stack = tk.ParamStackNode([node])

    def test_error_stack_stacks(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(3, 2),
                        axes_names=('input', 'output'),
                        network=net,
                        init_method='randn')
        node2 = tk.Node(shape=(3, 2),
                        axes_names=('input', 'output'),
                        network=net,
                        init_method='randn')

        # Create Stack with just one node
        stack = tk.StackNode([node1])
        with pytest.raises(TypeError):
            stack_stack = tk.StackNode([stack])
        with pytest.raises(TypeError):
            stack_stack = tk.ParamStackNode([stack])

        # Create Stack with two nodes
        stack = tk.StackNode([node1, node2])
        with pytest.raises(TypeError):
            stack_stack = tk.StackNode([stack])
        with pytest.raises(TypeError):
            stack_stack = tk.ParamStackNode([stack])

        # Create ParamStack with just one node
        stack = tk.ParamStackNode([node1])
        with pytest.raises(TypeError):
            stack_stack = tk.StackNode([stack])
        with pytest.raises(TypeError):
            stack_stack = tk.ParamStackNode([stack])

        # Create ParamStack with two nodes
        stack = tk.ParamStackNode([node1, node2])
        with pytest.raises(TypeError):
            stack_stack = tk.StackNode([stack])
        with pytest.raises(TypeError):
            stack_stack = tk.ParamStackNode([stack])

    def test_error_diff_type(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(3, 2),
                        axes_names=('input', 'output'),
                        network=net,
                        init_method='randn')
        node2 = tk.ParamNode(shape=(3, 2),
                             axes_names=('input', 'output'),
                             network=net,
                             init_method='randn')

        # Cannot stack Node's with ParamNode's
        with pytest.raises(TypeError):
            stack = tk.StackNode([node1, node2])
        with pytest.raises(TypeError):
            stack = tk.ParamStackNode([node1, node2])

    def test_error_diff_axes_names(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(3, 2),
                        axes_names=('input1', 'output1'),
                        network=net,
                        init_method='randn')
        node2 = tk.Node(shape=(3, 2),
                        axes_names=('input2', 'output2'),
                        network=net,
                        init_method='randn')

        # Cannot stack nodes with different axes names
        with pytest.raises(ValueError):
            stack = tk.StackNode([node1, node2])
        with pytest.raises(ValueError):
            stack = tk.ParamStackNode([node1, node2])

    def test_error_diff_network(self):
        node1 = tk.Node(shape=(3, 2),
                        axes_names=('input', 'output'),
                        init_method='randn')
        node2 = tk.Node(shape=(3, 2),
                        axes_names=('input', 'output'),
                        init_method='randn')

        assert node1.network != node2.network

        # Cannot stack nodes in different networks
        with pytest.raises(ValueError):
            stack = tk.StackNode([node1, node2])
        with pytest.raises(ValueError):
            stack = tk.ParamStackNode([node1, node2])

    def test_stack_change_stack_name(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 2),
                                axes_names=('input', 'output'),
                                name='node',
                                network=net,
                                init_method='randn')
            nodes.append(node)

        stack = tk.ParamStackNode(nodes)
        assert stack.axes_names == ['stack', 'input', 'output']

        # Name of stack edge cannot be changed
        with pytest.raises(ValueError):
            stack.get_axis('stack').name = 'other_name'


class TestTensorNetwork:

    def test_add_remove(self):
        net = tk.TensorNetwork()
        for _ in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]

        new_node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node')
        new_node.network = net
        assert new_node.name == 'node_4'

        assert len(net.edges) == 15
        for i in range(4):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.edges) == 7

        net._remove_node(new_node)
        # `new_node` still exists, but not in the network
        assert new_node.name not in net.nodes
        assert net['node_3']['right'].node2 == new_node

        new_node.network = net
        net.delete_node(new_node)
        # When deleting, we remove node AND disconnect,
        # so that we can now delete the object
        assert new_node.name not in net.nodes
        assert net['node_3']['right'].node2 is None
        assert new_node['left'].node2 is None
        del new_node

    def test_change_name(self):
        net = tk.TensorNetwork()
        for _ in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)
        node1 = net['node_0']
        node2 = net['node_1']

        node1.name = 'node_1'
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]
        assert node1.name == 'node_0'

        node1.name = 'my_node'
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(3)] \
            + ['my_node']
        assert node1.name == 'my_node'

        node2.name = 'my_node'
        assert list(net.nodes.keys()) == ['node_0', 'node_1',
                                          'my_node_0', 'my_node_1']
        assert node1.name == 'my_node_0'
        assert node2.name == 'my_node_1'

    def test_consecutive_contractions(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.edges) == 6

        node = net['node_0']
        for i in range(1, 4):
            node @= net[f'node_{i}']
        assert len(node.edges) == 6
        assert node.shape == (2, 5, 5, 5, 5, 2)
        assert len(net.nodes) == 7
        assert len(net.edges) == 6

        new_node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           init_method='randn')
        new_node.network = net
        assert new_node.name == 'node_4'
        net['node_3'][2] ^ new_node[0]
        with pytest.raises(ValueError):
            node @= new_node

    def test_submodules_empty(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             network=net)
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)

        # The network does not have submodules, ParamNodes are just
        # parameters of the network
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        # NOTE: This test was implemented when edges could be parametric,
        # Now the network never has submodules, only parameters
        net['paramnode_0']['right'] ^ net['paramnode_1']['left']
        net['paramnode_1']['right'] ^ net['node_0']['left']
        net['node_0']['right'] ^ net['node_1']['left']
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

    def test_submodules(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             network=net,
                             init_method='randn')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        init_method='randn')

        # The network does not have submodules, ParamNodes are just
        # parameters of the network
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 2

        # NOTE: This test was implemented when edges could be parametric
        # Now the network never has submodules, only parameters
        net['paramnode_0']['right'] ^ net['paramnode_1']['left']
        net['paramnode_1']['right'] ^ net['node_0']['left']
        net['node_0']['right'] ^ net['node_1']['left']
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 2

    def test_parameterize_tn_empty(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)

        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        param_net = net.parameterize()
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        submodules = [None for _ in param_net.children()]
        assert len(submodules) == 0
        assert len(param_net._parameters) == 0

        param_net = net.parameterize(override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        net.parameterize(set_param=False, override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

    def test_parameterize_tn(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        init_method='randn')

        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        param_net = net.parameterize()
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        submodules = [None for _ in param_net.children()]
        assert len(submodules) == 0
        assert len(param_net._parameters) == 2

        param_net = net.parameterize(override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 2

        net.parameterize(set_param=False, override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

    def test_set_data_nodes_same_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        assert len(net.data_nodes) == 0
        input_edges = []
        for i in range(4):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, 1)

        # 4 leaf nodes, 4 data nodes, and the
        # stack_data_memory
        assert len(net.nodes) == 9
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 4
        assert len(net.virtual_nodes) == 1

        input_edges = []
        for i in range(3):
            input_edges.append(net[f'node_{i}']['input'])
        with pytest.raises(ValueError):
            net.set_data_nodes(input_edges, 1)

        net.unset_data_nodes()
        assert len(net.nodes) == 4
        assert len(net.data_nodes) == 0

        input_edges = []
        for i in range(2):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, 1)
        assert len(net.nodes) == 7
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 2
        assert len(net.virtual_nodes) == 1

    def test_add_data_same_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        input_edges = []
        for i in range(2):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, 1)

        # data shape = batch_dim x n_features x feature_dim
        data = torch.randn(10, 2, 5)
        net.add_data(data)
        assert torch.equal(net.data_nodes['data_0'].tensor, data[:, 0, :])
        assert torch.equal(net.data_nodes['data_1'].tensor, data[:, 1, :])
        assert torch.equal(net.nodes['stack_data_memory'].tensor,
                           data.movedim(-2, 0))
        
        assert net.data_nodes['data_0'].shape == (10, 5)
        assert net.data_nodes['data_1'].shape == (10, 5)
        assert net.nodes['stack_data_memory'].shape == (2, 10, 5)

        # This causes no error, because the data tensor will be cropped to fit
        # the shape of the stack_data_memory node. It gives a warning
        data = torch.randn(10, 3, 5)
        net.add_data(data)
        
        assert net.data_nodes['data_0'].shape == (10, 5)
        assert net.data_nodes['data_1'].shape == (10, 5)
        assert net.nodes['stack_data_memory'].shape == (2, 10, 5)

        # This does not give warning, batch size can be changed as we wish
        data = torch.randn(100, 2, 5)
        net.add_data(data)
        
        assert net.data_nodes['data_0'].shape == (100, 5)
        assert net.data_nodes['data_1'].shape == (100, 5)
        assert net.nodes['stack_data_memory'].shape == (2, 100, 5)

        # Add data with no data nodes raises error
        net.unset_data_nodes()
        with pytest.raises(ValueError):
            net.add_data(data)

    def test_set_data_nodes_diff_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            node = tk.Node(shape=(2, i + 2, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='ones')
            data_node = tk.Node(shape=(i + 2, 1),  # Shape of batch index is irrelevant
                                axes_names=('feature', 'batch'),
                                name='data_node',
                                network=net,
                                init_method='ones',
                                data=True)
            node['input'] ^ data_node['feature']

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        assert len(net.nodes) == 8
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 4
        assert len(net.virtual_nodes) == 0

        net.unset_data_nodes()
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 0
        assert len(net.virtual_nodes) == 0

    def test_add_data_diff_shape(self):
        net = tk.TensorNetwork(name='net')
        data = []
        for i in range(4):
            node = tk.Node(shape=(2, i + 2, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='ones')
            data_node = tk.Node(shape=(i + 2, 1),  # Shape of batch index is irrelevant
                                axes_names=('feature', 'batch'),
                                name='data',
                                network=net,
                                init_method='ones',
                                data=True)
            node['input'] ^ data_node['feature']
            data.append(torch.randn(i + 2, 100))

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        net.add_data(data)
        for i in range(4):
            assert torch.equal(net.data_nodes[f'data_{i}'].tensor, data[i])
            assert net.data_nodes[f'data_{i}'].shape == data[i].shape

        # This does not raise error because indexing data[i] works
        # the same for lists or tensors where the "node" index is the
        # first dimension. Besides, the "feature" dimension has to be
        # greater than any of the "feature" dimensions used in data nodes,
        # since we are cropping
        data = torch.randn(4, 6, 100)
        net.add_data(data)

        # If feature dimension is small, it would raise an error
        data = torch.randn(4, 4, 100)
        with pytest.raises(ValueError):
            net.add_data(data)

        # Add data with no data nodes raises error
        net.unset_data_nodes()
        with pytest.raises(ValueError):
            net.add_data(data)

    def test_copy_tn(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        copy_net = net.copy()
        assert copy_net != net
        assert copy_net.nodes != net.nodes
        assert copy_net.edges != net.edges

        assert len(net.nodes) == 4
        assert len(net.edges) == 6

        assert len(copy_net.nodes) == 4
        assert len(copy_net.edges) == 6

    def test_delete_nodes(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6

        net.delete_node(net['node_3'])
        assert len(net.nodes) == 3
        assert len(net.edges) == 5

        node = net['node_0']
        for i in range(1, 3):
            node @= net[f'node_{i}']
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 3
        assert len(net.resultant_nodes) == 2
        assert len(net.edges) == 5

        net.delete_node(node)
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 3
        assert len(net.resultant_nodes) == 1
        assert len(net.edges) == 5
        
    def test_inverse_memory(self):
        net = tk.TensorNetwork()
        nodes = []
        for i in range(4):
            node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)

        for i in range(3):
            nodes[i]['right'] ^ nodes[i + 1]['left']
            
        input_edges = []
        for node in nodes:
            input_edges.append(node['input'])
            
        net.set_data_nodes(input_edges, 1)
        
        # Trace
        net._tracing = True
        net.add_data(torch.randn(100, 4, 5))
        aux_nodes = [node @ node.neighbours('input') for node in nodes]
        result1 = aux_nodes[0] @ aux_nodes[1]
        result2 = aux_nodes[2] @ aux_nodes[3]
        result = result1 @ result2
        
        # Repeat operations
        net._tracing = False
        net.add_data(torch.randn(100, 4, 5))
        aux_nodes = [node @ node.neighbours('input') for node in nodes]
        
        # Memory in nodes should be still there
        for node in nodes:
            assert node.tensor is not None
        
        # Memory in data nodes should be consumed:
        for data_node in list(net.data_nodes.values()):
            assert data_node.tensor is None
            assert net['stack_data_memory'].tensor is None
        
        result1 = aux_nodes[0] @ aux_nodes[1]
        assert aux_nodes[0].tensor is None
        assert aux_nodes[1].tensor is None
        
        result2 = aux_nodes[2] @ aux_nodes[3]
        assert aux_nodes[2].tensor is None
        assert aux_nodes[3].tensor is None
        
        result = result1 @ result2
        assert result1.tensor is None
        assert result2.tensor is None
        
    def test_inverse_memory_contract_data_stack(self):
        net = tk.TensorNetwork()
        nodes = []
        for i in range(4):
            node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)

        for i in range(3):
            nodes[i]['right'] ^ nodes[i + 1]['left']
            
        input_edges = []
        for node in nodes:
            input_edges.append(node['input'])
            
        net.set_data_nodes(input_edges, 1)
        
        # Trace
        net._tracing = True
        net.add_data(torch.randn(100, 4, 5))
        
        stack_nodes = tk.stack(nodes)
        stack_data = tk.stack(list(net.data_nodes.values()))
        stack_nodes['input'] ^ stack_data['feature']
        aux_nodes = tk.unbind(stack_nodes @ stack_data)
        
        result1 = aux_nodes[0] @ aux_nodes[1]
        result2 = aux_nodes[2] @ aux_nodes[3]
        result = result1 @ result2
        
        # Repeat operations
        net._tracing = False
        net.add_data(torch.randn(100, 4, 5))
        
        stack_nodes = tk.stack(nodes)
        stack_data = tk.stack(list(net.data_nodes.values()))
        stack_nodes['input'] ^ stack_data['feature']
        aux_nodes = tk.unbind(stack_nodes @ stack_data)
        
        # Memory in nodes should be still there
        for node in nodes:
            assert node.tensor is not None
        
        # Memory in data nodes should be consumed:
        for data_node in list(net.data_nodes.values()):
            assert data_node.tensor is None
            assert net['stack_data_memory'].tensor is None
        
        result1 = aux_nodes[0] @ aux_nodes[1]
        assert aux_nodes[0].tensor is None
        assert aux_nodes[1].tensor is None
        
        result2 = aux_nodes[2] @ aux_nodes[3]
        assert aux_nodes[2].tensor is None
        assert aux_nodes[3].tensor is None
        
        result = result1 @ result2
        assert result1.tensor is None
        assert result2.tensor is None
        
    def test_inverse_memory_contract_data_stack_auto_stack(self):
        net = tk.TensorNetwork()
        nodes = []
        for i in range(4):
            node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)

        for i in range(3):
            nodes[i]['right'] ^ nodes[i + 1]['left']
            
        input_edges = []
        for node in nodes:
            input_edges.append(node['input'])
            
        net.set_data_nodes(input_edges, 1)
        net.auto_stack = True
        
        # Trace
        net._tracing = True
        net.add_data(torch.randn(100, 4, 5))
        
        stack_nodes = tk.stack(nodes)
        stack_data = tk.stack(list(net.data_nodes.values()))
        stack_nodes['input'] ^ stack_data['feature']
        aux_nodes = tk.unbind(stack_nodes @ stack_data)
        
        result1 = aux_nodes[0] @ aux_nodes[1]
        result2 = aux_nodes[2] @ aux_nodes[3]
        result = result1 @ result2
        
        # Repeat operations
        net._tracing = False
        net.add_data(torch.randn(100, 4, 5))
        
        stack_nodes = tk.stack(nodes)
        stack_data = tk.stack(list(net.data_nodes.values()))
        stack_nodes['input'] ^ stack_data['feature']
        aux_nodes = tk.unbind(stack_nodes @ stack_data)
        
        # Memory in nodes should be still there
        for node in nodes:
            assert node.tensor is not None
        
        # Memory in data nodes should be consumed:
        for data_node in list(net.data_nodes.values()):
            assert data_node.tensor is None
            assert net['stack_data_memory'].tensor is None
        
        result1 = aux_nodes[0] @ aux_nodes[1]
        assert aux_nodes[0].tensor is None
        assert aux_nodes[1].tensor is None
        
        result2 = aux_nodes[2] @ aux_nodes[3]
        assert aux_nodes[2].tensor is None
        assert aux_nodes[3].tensor is None
        
        result = result1 @ result2
        assert result1.tensor is None
        assert result2.tensor is None
        

    def test_reset(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6

        stack_node1 = tk.stack([net['node_0'], net['node_2']])
        stack_node2 = tk.stack([net['node_1'], net['node_3']])
        stack_node1['right'] ^ stack_node2['left']
        assert len(net.nodes) == 6
        assert len(net.leaf_nodes) == 4
        assert len(net.resultant_nodes) == 2
        assert len(net.edges) == 6

        contract_node = stack_node1 @ stack_node2
        assert len(net.nodes) == 7
        assert len(net.leaf_nodes) == 4
        assert len(net.resultant_nodes) == 3
        assert len(net.edges) == 6

        net.reset()
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 4
        assert len(net.resultant_nodes) == 0
        assert len(net.edges) == 6
        
    def test_trace(self):
        pass

    def test_auto_stack(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        for node in net.nodes.values():
            assert node._tensor_info['address'] is not None

        assert net.auto_stack == False
        assert net.auto_unbind == False

        stack = tk.stack(list(net.nodes.values()))
        # All nodes still have their own memory
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
        for node in net.resultant_nodes.values():
            assert node._tensor_info['address'] is not None

        net.auto_stack = True
        assert net.auto_stack == True
        assert net.auto_unbind == False

        stack = tk.stack(list(net.nodes.values()))
        # Now leaf nodes have their emory stored in the stack
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is None
        for node in net.resultant_nodes.values():
            assert node._tensor_info['address'] is not None

    def test_auto_unbind(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        for node in net.nodes.values():
            assert node._tensor_info['address'] is not None

        assert net.auto_stack == False
        assert net.auto_unbind == False

        net.auto_unbind = True
        assert net.auto_stack == False
        assert net.auto_unbind == True
        
        stack1 = tk.stack(list(net.nodes.values()))
        unbinded = tk.unbind(stack1)
        stack2 = tk.stack(unbinded)
        # Now leaf nodes have their emory stored in the stack
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            assert node._tensor_info['node_ref'] is None

        # Unbinded nodes and `stack2` share memory with `stack1`
        assert stack1._tensor_info['address'] is not None
        for node in unbinded:
            assert node._tensor_info['address'] is None
        assert stack2._tensor_info['address'] is None

        net.auto_unbind = False
        assert net.auto_stack == False
        assert net.auto_unbind == False

        stack1 = tk.stack(list(net.nodes.values()))
        unbinded = tk.unbind(stack1)
        stack2 = tk.stack(unbinded)
        # All nodes still have their own memory
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            assert node._tensor_info['node_ref'] is None
        for node in net.resultant_nodes.values():
            assert node._tensor_info['node_ref'] is None
            assert node._tensor_info['address'] is not None
