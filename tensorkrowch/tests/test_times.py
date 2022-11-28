"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

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
        parents = [tk.randn(shape=(3, 4)) for _ in range(100)]
        op = 'stack'
        child = tk.randn(shape=(10, 3, 4))

    succ_obj = succesor()
    print()
    start = time.time()
    print(succ_obj.parents, succ_obj.op, succ_obj.child)
    print(time.time() - start)
    print('=' * 200)

    succ_dict = {'parents': [tk.randn(shape=(3, 4)) for _ in range(100)],
                 'op': 'stack',
                 'child': tk.randn(shape=(10, 3, 4))}
    print()
    start = time.time()
    print(succ_obj.parents, succ_obj.op, succ_obj.child)
    print(time.time() - start)
    print('=' * 200)
    # Almost the same


# TODO: remove later
def test_time_contraction_methods():
    node1 = tk.Node(shape=(100, 10, 100, 10), name='node1', param_edges=True, init_method='randn')
    node2 = tk.Node(shape=(100, 10, 100, 10), name='node2', param_edges=True, init_method='randn')
    node1[0] ^ node2[0]
    node1[2] ^ node2[2]

    print()
    start = time.time()
    node3 = node1 @ node2
    print(time.time() - start)
    # Prev contraction method: 0.03
    # New contraction method: 0.0045
    # Mucho mejor el nuevo

    node1 = tk.Node(shape=(100, 10, 100, 10), name='node1', init_method='randn')
    node2 = tk.Node(shape=(100, 10, 100, 10), name='node2', init_method='randn')
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
    s = tk.stack_unequal_tensors(lst)
    print(time.time() - start)

    lst = [torch.randn(torch.randint(low=1, high=100, size=(1,)).long().item(), 100, 100) for _ in range(1000)]
    start = time.time()
    s = tk.stack_unequal_tensors(lst)
    print(time.time() - start)


def test_time_check_kwargs():
    print()
    lst = [tk.randn((20, 30)) for _ in range(100)]
    h1 = hash(tuple(lst))

    lst2 = [tk.randn((20, 30)) for _ in range(100)]
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

    # Triangular as sparse
    a_sparse = a.to_sparse_coo()
    start = time.time()
    c = torch.bmm(a_sparse, b)
    print('\nTriangular as sparse:')
    print(time.time() - start)
    memory = a_sparse.indices().nelement() * a_sparse.indices().element_size() + \
             a_sparse.values().nelement() * a_sparse.values().element_size()
    print(f'{memory / 1024**2:.2f} Mb')  # It's worse than dense tensor

    class TriMat:
        def __init__(self, trimat):
            assert trimat.shape[-2] == trimat.shape[-1]  # Asumimos primero que son cuadradas
            lst = []
            for i in range(trimat.shape[-2]):
                lst.append(trimat[..., i, :(i+1)])
            self.lst = lst
            self.shape = trimat.shape

        def as_tensor(self):
            aux_lst = []
            for i, tensor in enumerate(self.lst):
                tensor = nn.functional.pad(tensor, (0, 100 - (i+1)))
                aux_lst.append(tensor)
            result = torch.stack(aux_lst, dim=1)
            return result

    # Handmade multiplications
    start_total = time.time()
    c = torch.zeros(1000, 100, 100)
    for i in range(100):
        c[:, i, :] = (a[:, i, :].unsqueeze(-1) * b).sum(dim=-2)
    print('\nRandom handmade mul.:')
    print(time.time() - start_total)

    a = TriMat(a)
    b = torch.randn(1000, 100, 100)
    start_total = time.time()
    c = torch.zeros(1000, 100, 100)
    for i in range(100):
        c[:, i, :] = (a.lst[i].unsqueeze(-1) * b[:, :(i+1), :]).sum(dim=-2)
    print('\nTriangular handmade mul.:')
    print(time.time() - start_total)

    a_aux = a.as_tensor()
    c2 = a_aux @ b  # TODO: al usar mi stack estaba modificando la lista de tensores que paso
    assert torch.allclose(c, c2, rtol=1e-3, atol=1e-5)

    print('\nTriangular as TriMat:')
    s = 0
    for t in a.lst:
        s += t.element_size() * t.nelement()
    print(f'{s / 1024**2:.2f} Mb')

    # Handmade option 2
    a = torch.randn(1000, 128, 128)
    b = torch.randn(1000, 128, 128)

    start = time.time()
    c = a @ b
    print('\nRandom 2:')
    print(time.time() - start)

    def foo(a, b, c):
        shape = a.shape[1]
        half_shape = shape // 2
        if half_shape > 1:
            c[:, :half_shape, half_shape:] = a[:, :half_shape, half_shape:] @ b[:, :half_shape, half_shape:]
            c[:, half_shape:, :half_shape] = a[:, half_shape:, :half_shape] @ b[:, half_shape:, :half_shape]
            c[:, :half_shape, :half_shape] = foo(a[:, :half_shape, :half_shape],
                                                 b[:, :half_shape, :half_shape],
                                                 c[:, :half_shape, :half_shape])
            c[:, half_shape:, half_shape:] = foo(a[:, half_shape:, half_shape:],
                                                 b[:, half_shape:, half_shape:],
                                                 c[:, half_shape:, half_shape:])
        else:
            c[:, :, :] = a @ b
        return c
    start_total = time.time()
    c = torch.zeros(1000, 128, 128)
    c = foo(a, b, c)
    print('\nRandom handmade mul. 2:')
    print(time.time() - start_total)

    def foo(a, b, c):
        shape = a.shape[1]
        half_shape = shape // 2
        if half_shape > 1:
            c[:, half_shape:, :half_shape] = a[:, half_shape:, :half_shape] @ b[:, half_shape:, :half_shape]
            c[:, :half_shape, :half_shape] = foo(a[:, :half_shape, :half_shape],
                                                 b[:, :half_shape, :half_shape],
                                                 c[:, :half_shape, :half_shape])
            c[:, half_shape:, half_shape:] = foo(a[:, half_shape:, half_shape:],
                                                 b[:, half_shape:, half_shape:],
                                                 c[:, half_shape:, half_shape:])
        else:
            c[:, :, :] = a @ b
        return c
    start_total = time.time()
    c = torch.zeros(1000, 128, 128)
    c = foo(a, b, c)
    print('\nTriangular handmade mul. 2:')
    print(time.time() - start_total)

    # # from functools import partial
    # # @partial(torch.jit.trace, example_inputs=(a, b))
    # # @torch.jit.script
    # def foo(mat1, mat2):
    #     res = torch.zeros(1000, 100, 100)
    #     for i in range(100):
    #         res[:, i, :] = (mat1[:, i, :].unsqueeze(-1) * mat2).sum(dim=-2)
    #     return res
    #
    # foo = torch.jit.trace(foo, (a, b))
    #
    # start_total = time.time()
    # c = foo(a, b)
    # print('\nRandom handmade mul. traced:')
    # print(time.time() - start_total)

    # model = nn.Linear(100, 100)
    # start_total = time.time()
    # c = model(a[..., 0])
    # print('\nmodel 1:')
    # print(time.time() - start_total)
    #
    # model = torch.jit.trace(nn.Linear(100, 100), torch.zeros(1000, 100))
    # start_total = time.time()
    # c = model(a[..., 0])
    # print('\nmodel 2:')
    # print(time.time() - start_total)

    # def foo(mat1, mat2): return mat1 @ mat2
    # start_total = time.time()
    # c = foo(a, b)
    # print('\nfoo 1:')
    # print(time.time() - start_total)
    #
    # foo = torch.jit.script(foo)
    # start_total = time.time()
    # c = foo(a, b)
    # print('\nfoo 2:')
    # print(time.time() - start_total)
    # start_total = time.time()
    # c = foo(a, b)
    # print('\nfoo 2:')
    # print(time.time() - start_total)


def test_scipy():
    # TODO: Esto podr'ia funcionar, pero me tengo que conformar con usar solo matrices
    print()
    from scipy.sparse import csr_array, triu, tril

    # Memory
    a = csr_array(torch.randn(10, 10).numpy())
    a_memory = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
    print(f'Memory of a: {a_memory/1024**2:.2f} Mb')

    b = triu(a).tocsr()
    b_memory = b.data.nbytes + b.indptr.nbytes + b.indices.nbytes
    print(f'Memory of b: {b_memory/1024**2:.2f} Mb')

    c = tril(a).tocsr()
    c_memory = c.data.nbytes + c.indptr.nbytes + c.indices.nbytes
    print(f'Memory of c: {c_memory/1024**2:.2f} Mb')

    # Multiplication time
    start = time.time()
    res = a.multiply(a)
    print('a @ a:', time.time() - start)

    start = time.time()
    res = b.multiply(b)
    print('b @ b:', time.time() - start)

    start = time.time()
    res = c.multiply(c)
    print('c @ c:', time.time() - start)

    start = time.time()
    res = a.multiply(b)
    print('a @ b:', time.time() - start)

    start = time.time()
    res = b.multiply(a)
    print('b @ a:', time.time() - start)

    start = time.time()
    res = a.multiply(c)
    print('a @ c:', time.time() - start)

    start = time.time()
    res = c.multiply(a)
    print('c @ a:', time.time() - start)

    start = time.time()
    res = c.multiply(b)
    print('c @ b:', time.time() - start)

    start = time.time()
    res = b.multiply(c)
    print('b @ c:', time.time() - start)

    # Compare matrix vs tensor
    import numpy as np

    tensor1 = torch.randn(1000, 200, 300).triu()
    tensor2 = torch.randn(1000, 300, 100).triu()

    sparse_mat1 = csr_array(tensor1.reshape(-1, 300).numpy())
    sparse_mat2 = csr_array(tensor2.permute(1, 2, 0).reshape(300, -1))

    start = time.time()
    # res = sparse_mat1.multiply(sparse_mat2)

    a_memory = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
    print(f'Memory of a: {a_memory / 1024 ** 2:.2f} Mb')


def test_times_view_reshape():
    a = torch.randn(100, 200, 300)
    print()
    
    view = []
    for _ in range(10000):
        start = time.time()
        a = a.view(100, 1, 200, 300)
        view.append(time.time() - start)
        a = a.view(100, 200, 300)
    print('View:', torch.tensor(view).mean())
        
    reshape = []  
    for _ in range(10000):
        start = time.time()
        a = a.reshape(100, 1, 200, 300)
        reshape.append(time.time() - start)
        a = a.reshape(100, 200, 300)
    print('Reshape:', torch.tensor(reshape).mean())
        
    # Both alternatives take almost the same amount of time
    print()
