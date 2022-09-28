"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn

import time
import opt_einsum
import dis


# TODO: remove test - check times if using bmm instead of einsum
def test_einsum_time():
    a = torch.randn(20, 30, 40)
    b = torch.randn(30, 60, 40)
    c = torch.randn(20, 60)
    print()
    print('Start')
    for _ in range(10):
        start = time.time()
        d = opt_einsum.contract('ijk,jlk,il->', a, b, c)
        print(time.time() - start)
    print()
    print('Start')
    for _ in range(10):
        start = time.time()
        aux = a.reshape(20, -1) @ b.permute(0, 2, 1).reshape(-1, 60)
        d = aux.flatten() @ c.flatten()
        print(time.time() - start)
    # Mucho más rápido!!

    a = torch.randn(20, 30)
    b = torch.randn(30, 40)
    print()
    print('Start')
    for _ in range(10):
        start = time.time()
        c = opt_einsum.contract('ij,jk->', a, b)
        print(time.time() - start)
    print()
    print('Start')
    for _ in range(10):
        start = time.time()
        c = a @ b
        print(time.time() - start)
    # Mucho más rápido!!


# TODO: remove later
def test_dis():
    def foo(x: int, y: int):
        return x + 1
    def bar(z: int):
        w = foo(z, z)
        return w
    print()
    print(dis.dis(bar))
    bytecode = dis.Bytecode(bar)
    funcs = []
    instrs = list(reversed([instr for instr in bytecode]))
    for (ix, instr) in enumerate(instrs):
        if instr.opname == "CALL_FUNCTION":
            load_func_instr = instrs[ix + instr.arg + 1]
            funcs.append(load_func_instr.argval)
    funcs


# TODO: remove later
def test_time_unbind():
    print()
    s = torch.randn(1000, 100, 100)

    start = time.time()
    even = s[0:1000:2]
    odd = s[1:1000:2]
    result = even @ odd
    print('Indexing stack:', time.time() - start)

    start = time.time()
    lst = s.unbind(0)
    even = torch.stack(lst[0:1000:2])
    odd = torch.stack(lst[1:1000:2])
    result = even @ odd
    print('Unbinding stack:', time.time() - start)
    # Efectivamente, caca, como el triple de tiempo


# TODO: remove later
def test_define_class():
    class succesor:
        parents = [tn.randn(shape=(3, 4)) for _ in range(100)]
        op = 'stack'
        child = tn.randn(shape=(10, 3, 4))

    succ_obj = succesor()
    print()
    start = time.time()
    print(succ_obj.parents, succ_obj.op, succ_obj.child)
    print(time.time() - start)
    print('=' * 200)

    succ_dict = {'parents': [tn.randn(shape=(3, 4)) for _ in range(100)],
                 'op': 'stack',
                 'child': tn.randn(shape=(10, 3, 4))}
    print()
    start = time.time()
    print(succ_obj.parents, succ_obj.op, succ_obj.child)
    print(time.time() - start)
    print('=' * 200)
    # Almost the same


# TODO: remove later
def test_time_contraction_methods():
    node1 = tn.Node(shape=(100, 10, 100, 10), name='node1', param_edges=True, init_method='randn')
    node2 = tn.Node(shape=(100, 10, 100, 10), name='node2', param_edges=True, init_method='randn')
    node1[0] ^ node2[0]
    node1[2] ^ node2[2]

    print()
    start = time.time()
    node3 = node1 @ node2
    print(time.time() - start)
    # Prev contraction method: 0.03
    # New contraction method: 0.0045
    # Mucho mejor el nuevo

    node1 = tn.Node(shape=(100, 10, 100, 10), name='node1', init_method='randn')
    node2 = tn.Node(shape=(100, 10, 100, 10), name='node2', init_method='randn')
    node1[0] ^ node2[0]
    node1[2] ^ node2[2]

    print()
    start = time.time()
    node3 = node1 @ node2
    print(time.time() - start)
    # Prev contraction method: 0.0015
    # New contraction method: 0.0015
    # Prácticamente iguales


# TODO: time padding
def test_time_padding():
    t = torch.randn(100, 100)
    print()

    start = time.time()
    t2 = torch.zeros(200, 200)
    t2[100:, 100:] = t
    print(time.time() - start)

    start = time.time()
    t2 = nn.functional.pad(t, (100, 0, 100, 0))
    print(time.time() - start)
    # Padding m'as r'apido claramente (un orden de magnitud aprox.)