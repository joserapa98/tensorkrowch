import tentorch as tn

node1 = tn.ParamNode(shape=(2, 5, 3),
                     axes_names=('left', 'input', 'right'),
                     name='node1',
                     network=None,
                     param_edges=False,
                     tensor=None,
                     init_method='ones')

node2 = tn.ParamNode(shape=(3, 5, 4),
                     axes_names=('left', 'input', 'right'),
                     name='node2',
                     network=None,
                     param_edges=False,
                     tensor=None,
                     init_method='ones')

print(node1.edges)
print(node2.edges)
print()

node1['right'] ^ node2['left']
print(node1.edges)
print(node2.edges)

node3 = tn.batched_contract_between(node1, node2, node1['input'], node2['input'])
print(node3)
print(node3.axes_names)
print(node3.edges)
