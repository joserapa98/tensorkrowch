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
    print('Index stack:', time.time() - start)

    start = time.time()
    lst = s.unbind(0)
    even = torch.stack(lst[0:1000:2])
    odd = torch.stack(lst[1:1000:2])
    result = even @ odd
    print('Unbind stack, index list, create new stacks:', time.time() - start)
    # Efectivamente, caca

    idx1 = torch.arange(0, 1000, 2)
    idx2 = torch.arange(1, 1000, 2)

    start = time.time()
    even = s[idx1]
    odd = s[idx2]
    result = even @ odd
    print('Create index (not slice), index stack:', time.time() - start)
    # Tarda m'as que usando slice

    start = time.time()
    lst = s.unbind(0)
    aux_lst = []
    for i in idx1:
        aux_lst.append(lst[i])
    even = torch.stack(aux_lst)
    aux_lst = []
    for i in idx2:
        aux_lst.append(lst[i])
    odd = torch.stack(aux_lst)
    result = even @ odd
    print('Create index, unbind stack, select from list, create stack:', time.time() - start)
    # Tarda lo mismo que usando slice


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


def test_time_check_equal_lists():
    t1 = []
    for _ in range(1000):
        t1.append(torch.randn(100, 100))

    t2 = t1[:]

    t1 = tuple(t1)
    t2 = tuple(t2)

    print()
    start = time.time()
    b = t1 == t2
    print('Classic check:', time.time() - start)

    start = time.time()
    b = hash(t1) == hash(t2)
    print('Hash check:', time.time() - start)
    # Mucho peor!


def test_time_stacks():
    print()
    lst = [torch.randn(100, 100, 100) for _ in range(1000)]
    start = time.time()
    s = torch.stack(lst)
    print(time.time() - start)

    lst = [torch.randn(100, 100, 100) for _ in range(1000)]
    start = time.time()
    s = tn.stack_unequal_tensors(lst)
    print(time.time() - start)

    lst = [torch.randn(torch.randint(low=1, high=100, size=(1,)).long().item(), 100, 100) for _ in range(1000)]
    start = time.time()
    s = tn.stack_unequal_tensors(lst)
    print(time.time() - start)


def test_time_check_kwargs():
    print()
    lst = [tn.randn((20, 30)) for _ in range(100)]
    h1 = hash(tuple(lst))

    lst2 = [tn.randn((20, 30)) for _ in range(100)]
    h2 = hash(tuple(lst2))

    # Con listas
    start = time.time()
    print(lst == lst)
    print(time.time() - start)

    start = time.time()
    print(lst == lst2)
    print(time.time() - start)

    # Con hashes  ->  cuando es True, es un orden de magnitud más rápido
    start = time.time()
    print(h1 == h1)
    print(time.time() - start)

    start = time.time()
    print(h1 == h2)
    print(time.time() - start)
    # TODO: cuando es True, es un orden de magnitud más rápido -> implementar hashes para kwargs


def test_time_stack_vs_cat():
    print()
    lst = [torch.randn(100, 100, 100) for _ in range(100)]
    t = torch.randn(100, 100, 100)

    # Stack
    start = time.time()
    aux1 = torch.stack(lst + [t])
    print(time.time() - start)

    # Cat  ->  un poco menos, pero casi iguales
    s = torch.stack(lst)
    start = time.time()
    aux2 = torch.cat([s, t.view(1, *t.shape)])
    print(time.time() - start)

    assert torch.equal(aux1, aux2)


def test_time_index():
    print()
    # List
    t = torch.randn(100, 100, 100)
    idx = torch.randint(0, 100, (50,)).tolist()
    start = time.time()
    t = t[idx]
    print(time.time() - start)
    print(t.shape)

    # Slice
    t = torch.randn(100, 100, 100)
    idx = slice(0, 100, 2)
    start = time.time()
    t = t[idx]
    print(time.time() - start)
    print(t.shape)

    # List of one element
    t = torch.randn(100, 100, 100)
    idx = [50]
    start = time.time()
    t = t[idx]
    print(time.time() - start)
    print(t.shape)

    # Index one element
    t = torch.randn(100, 100, 100)
    idx = 50
    start = time.time()
    t = t[idx].view(1, *t.shape[1:])
    print(time.time() - start)
    print(t.shape)

    # Slice of one element
    t = torch.randn(100, 100, 100)
    idx = slice(50, 51)
    start = time.time()
    t = t[idx]
    print(time.time() - start)
    print(t.shape)


def test_time_bmm():
    print()
    # Matrix - Matrix
    m1 = torch.randn(500, 10, 10)
    m2 = torch.randn(500, 10, 10)
    start = time.time()
    m1 @ m2
    print(time.time() - start)

    # Matrix - Vector
    v = torch.randn(500, 1, 10)
    m = torch.randn(500, 10, 10)
    start = time.time()
    v @ m
    print(time.time() - start)


def test_time_select_backward():
    print()
    a = nn.Parameter(torch.randn(1000, 10, 10))

    # Unbind  ->  Más rápido a partir de batch = 500, antes más lento
    lst = a.unbind()
    start = time.time()
    result = lst[0]
    for i in range(1, 1000):
        result = result @ lst[i]
    result = result.sum()
    result.backward()
    print(time.time() - start)

    # Select
    lst = [a[i] for i in range(a.shape[0])]
    start = time.time()
    result = lst[0]
    for i in range(1, 1000):
        result = result @ lst[i]
    result = result.sum()
    result.backward()
    print(time.time() - start)


def test_time_stack_backward():
    print()
    a = nn.Parameter(torch.randn(1000, 10, 10))

    # Unbind + Stack
    lst = list(a.unbind())
    stack1 = torch.stack(lst[0:1000:2])
    stack2 = torch.stack(lst[1:1000:2])
    start = time.time()
    result = stack1 @ stack2
    result = result.sum()
    print('Forward:', time.time() - start)
    start = time.time()
    result.backward()
    print('Backward:', time.time() - start)

    # Slice
    stack1 = a[0:1000:2]
    stack2 = a[1:1000:2]
    start = time.time()
    result = stack1 @ stack2
    result = result.sum()
    print('Forward:', time.time() - start)
    start = time.time()
    result.backward()
    print('Backward:', time.time() - start)


def test_time_stay_the_same():
    a = torch.randn(100, 200, 300)

    # Permute
    start = time.time()
    a.permute(2, 1, 0)
    print('\nPermute diff.:', time.time() - start)

    start = time.time()
    a.permute(0, 1, 2)
    print('Permute same:', time.time() - start)

    start = time.time()
    if [0, 1, 2] == list(range(len(a.shape))):
        pass
    print('Permute same (no-op):', time.time() - start)

    start = time.time()
    if not []:
        pass
    print('Permute same (no-op, no-check):', time.time() - start)

    # Reshape
    start = time.time()
    a.reshape(-1)
    print('\nReshape diff.:', time.time() - start)

    start = time.time()
    a.reshape(100, 200, 300)
    print('Reshape same:', time.time() - start)

    start = time.time()
    if [100, 200, 300] == a.shape:
        pass
    print('Reshape same (no-op):', time.time() - start)

    # View
    start = time.time()
    a.view(-1)
    print('\nView diff.:', time.time() - start)

    start = time.time()
    a.view(100, 200, 300)
    print('View same:', time.time() - start)

    start = time.time()
    if [100, 200, 300] == a.shape:
        pass
    print('View same (no-op):', time.time() - start)


def test_memory():
    print()

    def same_storage(x, y):
        # TODO: We have replaced view with reshape, maybe this changes something
        x_ptrs = set(e.data_ptr() for e in x.reshape(-1))
        y_ptrs = set(e.data_ptr() for e in y.reshape(-1))
        return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)

    # Stack tensors
    lst = [torch.randn(2, 3) for _ in range(100)]
    stack = torch.stack(lst)  # Does allocate new memory
    print(same_storage(lst[0], stack))

    # Stack parameters
    lst = [nn.Parameter(torch.randn(2, 3)) for _ in range(10)]
    stack = torch.stack(lst)  # Does allocate new memory
    print(same_storage(lst[0], stack))

    assert lst[0][0, 0] == stack[0, 0, 0]
    optimizer = torch.optim.SGD(lst, lr=0.01)
    stack.sum().backward()
    optimizer.step()
    assert lst[0][0, 0] != stack[0, 0, 0]

    # Unbind tensors
    unbinded = stack.unbind()  # Does not allocate new memory
    print(same_storage(unbinded[0], stack))

    # Slice from stack
    sliced = stack[0:50]  # Does not allocate new memory
    print(same_storage(sliced, stack))

    # Restack unbinded tensors
    restack = torch.stack(unbinded)  # Allocates new memory again
    print(same_storage(restack, stack))

    # Permute
    permute = stack.permute(1, 0, 2)  # Allocates new memory
    print(same_storage(permute, stack))


def test_time_matmul_with_zeros():
    # Random tensors
    a = torch.randn(1000, 100, 100)
    b = torch.randn(1000, 100, 100)
    start = time.time()
    c = a @ b
    print('\nRandom:')
    print(time.time() - start)
    print(f'{a.element_size() * a.nelement() / 1024**2:.2f} Mb')

    # Triangular matrices
    a = torch.tril(a)
    b = torch.tril(b)
    start = time.time()
    c = a @ b
    print('\nTriangular:')
    print(time.time() - start)
    print(f'{a.element_size() * a.nelement() / 1024**2:.2f} Mb')

    aux = a.to_sparse_coo()
    print(f'{aux.element_size() * aux.nelement() / 1024 ** 2:.2f} Mb')

    class TriMat:
        def __init__(self, trimat):
            assert trimat.shape[-2] == trimat.shape[-1]  # Asumimos primero que son cuadradas
            lst = []
            for i in range(trimat.shape[-2]):
                lst.append(trimat[..., i, :(i+1)])
            self.lst = lst
            self.shape = trimat.shape

        def as_tensor(self):
            result = tn.stack_unequal_tensors(self.lst)
            return result.permute(1, 0, 2)

    a = TriMat(a)
    # b = torch.tril(b)
    # start = time.time()
    # c = torch.zeros(1000, 100, 100)
    # for i in range(100):
    #     for j in range(i+1):
    #         for k in range(100):
    #             c[:, i, k] = a.lst[i][:, j] * b[:, j, k]
    # print('\nTriangular optimized:')
    # print(time.time() - start)
    #
    # c2 = a.as_tensor() @ b  # TODO: al usar mi stack estaba modificando la lista de tensores que paso
    # assert torch.equal(c, c2)

    s = 0
    for t in a.lst:
        s += t.element_size() * t.nelement()
    print(f'{s / 1024**2:.2f} Mb')
