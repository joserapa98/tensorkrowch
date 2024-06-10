"""
This script contains:

    * extend_with_output
    * sketching
    * trimming
    * create_projector
    * tt_rss
"""

import warnings
from typing import Optional, Callable, List

import torch
from torch.utils.data import TensorDataset, DataLoader

from tensorkrowch.embeddings import basis
from tensorkrowch.utils import random_unitary


def extend_with_output(function, samples, labels, out_position, batch_size, device):
    """
    Extends ``samples`` tensor with the output of the ``function`` evaluated on
    those ``samples``. If ``samples`` is a tensor with shape ``batch_size x
    n_features``, the extended samples will have shape ``batch_size x
    (n_features + 1)``.
    
    NOTE: What if ``samples`` has shape ``batch_size x n_features x in_dim``?
    
    If ``labels`` are not provided, they are obtained by passing the ``samples``
    through the ``function``. In this case, it is assumed that ``function``
    returns a vector of probabilities (for each class). Thus, the vector of
    ``labels`` is sampled according to the output distribution.
    
    If ``labels`` are give, it is assumed to be a tensor with shape ``batch_size``.
    
    NOTE: What if ``samples`` has shape ``batch_size x n_features x in_dim``?
    How do we concatenate the samples and the labels?
    """
    if labels is None:
        loader = DataLoader(
            TensorDataset(samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
        
        with torch.no_grad():
            outputs = []
            for (batch,) in loader:
                outputs.append(function(batch.to(device)).pow(2).cpu())
            
            # NOTE: for relu or other functions, sample the label uniformly
            
            outputs = torch.cat(outputs, dim=0).cpu()
            
            labels_distr = outputs.cumsum(dim=1)
            labels_distr = labels_distr / labels_distr[:, -1:]
            
            probs = torch.rand(outputs.size(0), 1)
            ids = outputs.size(1) - torch.le(probs,
                                             labels_distr).sum(dim=1, keepdim=True)
            
            extended_samples = torch.cat([samples[:, :out_position],
                                          ids,
                                          samples[:, out_position:]], dim=1)
            return extended_samples, outputs.gather(dim=1, index=ids).pow(0.5)
    else:
        ids = labels.view(-1, 1)
        extended_samples = torch.cat([samples[:, :out_position],
                                      ids,
                                      samples[:, out_position:]], dim=1)
        return extended_samples, torch.ones_like(ids).float()


def sketching(function, tensors_list, out_position, batch_size, device):
    """
    Given 'tensors_list', where each tensor i has shape di x ni and
    sum(n1, ..., nk) = n_features = input_size, creates a projection tensor
    with shape d1 x ... x dk x n_features, and computes Phi_tilde_k of shape
    d1 x ... x dk
    """
    sizes = []
    for tensor in tensors_list:
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) == 2
        sizes.append(tensor.size(1))
    
    # Expand all tensors so that each one has shape d1 x ... x dk x ni
    aux_list = []
    for i in range(len(tensors_list)):
        view_shape = []
        expand_shape = []
        for j in range(len(tensors_list)):
            if j == i:
                view_shape.append(tensors_list[i].size(0))
                expand_shape.append(-1)
            else:
                view_shape.append(1)
                expand_shape.append(tensors_list[j].size(0))
        view_shape.append(tensors_list[i].size(1))
        expand_shape.append(-1)
        
        aux_tensor = tensors_list[i].view(*view_shape).expand(*expand_shape)
        aux_list.append(aux_tensor)
    
    # Find labels in out_position
    # TODO: I guess labels could be computed and passed from the begginings
    # without haveing to recover them in each sketching. Also, labels are always
    # the same, so yes
    if out_position >= 0: # If function is vector-valued
        cum_size = 0
        for i, size in enumerate(sizes):
            if (cum_size + size - 1) >= out_position:
                if size == 1:
                    labels = aux_list[i][..., 0]
                    aux_list = aux_list[:i] + aux_list[(i + 1):]
                else:
                    labels = aux_list[i][..., out_position - cum_size]
                    aux_list[i] = torch.cat(
                        [aux_list[i][..., :(out_position - cum_size)],
                        aux_list[i][..., (out_position - cum_size + 1):]], dim=-1)
                labels = labels.reshape(-1, 1).to(torch.int64).to(device)
                break
            cum_size += size
    else:
        labels = torch.zeros(*aux_list[0].shape[:-1], 1).to(torch.int64).to(device)
        labels = labels.view(-1, 1)
    
    projection = torch.cat(aux_list, dim=-1)
    
    projection_loader = DataLoader(
        TensorDataset(projection.view(-1, projection.shape[-1]), labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    
    Phi_tilde_k = []
    with torch.no_grad():
        for idx, (batch, labs) in enumerate(projection_loader):
            # print('batch:', idx)
            aux_result = function(batch.to(device))
            Phi_tilde_k.append(aux_result.gather(dim=1,
                                                 index=labs).flatten().cpu())
    Phi_tilde_k = torch.cat(Phi_tilde_k, dim=0)
            
    # with torch.no_grad():
    #     aux_result = function(projection.view(-1, projection.shape[-1]).to(device))
    #     Phi_tilde_k = aux_result.gather(dim=1, index=labels).flatten().cpu()
            
    Phi_tilde_k = Phi_tilde_k.view(*projection.shape[:-1])
    return Phi_tilde_k


def trimming(mat, max_D, cum_percentage):
    """Given a matrix returns the U from the SVD and an appropiate rank"""
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    
    if max_D is None:
        max_D = len(s)

    percentages = s.cumsum(0) / (s.sum().expand(s.shape) + 1e-10)
    cum_percentage_tensor = torch.tensor(cum_percentage)
    rank = 0
    for p in percentages:
        if p == 0:
            if rank == 0:
                rank = 1
            break
        rank += 1
        # Cut when ``cum_percentage`` is exceeded
        if p >= cum_percentage_tensor:
            break
        elif rank >= max_D:
            break
        
    return u, rank


def create_projector(S_k_1, S_k):
    """
    Given the previous projector and the current one, it infers the s_k needed
    to create S_k from S_k_1. All rows of S_k_1 and S_k must be unique!

    Parameters
    ----------
    S_k_1 : torch.Tensor
        Matrix of shape n x k. The n rows of S_k_1 are equal to the rows of
        S_k[:, :-1], but maybe they are repeated in S_k.
    S_k : torch.Tensor
        Matrix of shape m x (k + 1), with m > n.
    
    Example
    -------
    >>> S_k = torch.randint(low=0, high=2, size=(10, 4)).unique(dim=0)
    >>> S_k
    
    tensor([[0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1]])
            
    >>> S_k_1 = S_k[:, :-1].unique(dim=0)
    >>> S_k_1
    
    tensor([[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0]])
            
    >>> create_projector(S_k_1, S_k)
    
    tensor([[0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 1],
            [3, 0],
            [4, 0],
            [4, 1]])
    """
    s_k = []
    for i in range(S_k_1.size(0)):
        where_equal = (S_k[:, :-1] == S_k_1[i]).all(dim=1)
        new_col = S_k[where_equal, -1:]
        first_col = torch.Tensor([[i]]).expand(new_col.size(0), 1).to(new_col.device)
        s_k.append(torch.cat([first_col, new_col], dim=1))
        
    s_k = torch.cat(s_k, dim=0)
    return s_k  


def tt_rss(function: Callable,
           embedding: Callable,
           sketch_samples: torch.Tensor,
           labels: Optional[torch.Tensor] = None,
           out_position: Optional[int] = None,
           rank: Optional[int] = None,
           cum_percentage: Optional[float] = None,
           batch_size: int = 64,
           device: Optional[torch.device] = None,
           verbose: bool = True) -> List[torch.Tensor]:
    r"""
    NOTE: we are assuming the function to decompose is given as square roots of
    probabilities. Outputs are sampled according to the distribution given
    as the squares of the outputs. What happens when the outputs do not represent
    probabilities? Do we sample outputs according to distribution? Or just uniformly?
    
    Tensor Train - Recursive Sketching from Samples.
    
    Decomposes a scalar function of :math:`N` input variables in a Matrix
    Product State of :math:`N` cores, each corresponding to one input variable,
    in the same order as they are provided to the function. To turn each input
    variable into a vector that can be contracted with the corresponding MPS
    core, an embedding function is required. The dimension of the embedding
    will be used as the input dimension of the MPS.
    
    If the function is vector-valued, it will be seen as a :math:`N + 1` scalar
    function, returning a MPS with :math:`N + 1` cores. The output variable will
    use the embedding :func:`~tensorkrowch.basis`, which maps integers
    (corresponding to indices of the output vector) to basis vectors:
    :math:`i \mapsto \langle i \rvert`. It can be specified the position in
    which the output core will be. By default, it will be in the middle of the
    MPS.
    
    When MPS cores are created, its rank has to be inferred. To control the
    maximum bond dimension for each core, we can specify one or both arguments
    ``max_bond_dim``, ``cum_percentage``.
    
    To specify the bond dimension of each MPS core, one can use the arguments
    ``rank`` and ``cum_percentage``. If more than one is specified, the
    resulting rank will be the one that satisfies all conditions.
    
    Parameters
    ----------
    function : Callable
        Function that is going to be decomposed. It needs to have a single
        input argument, the data, which is a tensor of shape
        ``batch_size x n_features``. It must return a tensor of shape
        ``batch_size x out_dim``. If the function is scalar, ``out_dim = 1``.
    embedding : Callable
        Embedding function that maps the data tensor to a higher dimensional
        space. It needs to have a single argument. It is a function that
        transforms the given data tensor and returns an embedded tensor of
        shape ``batch_size x n_features x in_dim``.
    sketch_samples : torch.Tensor
        Samples that will be used as sketches to decompose the function. It has
        to be a tensor of shape ``batch_size x n_features`` or
        ``batch_size x n_features x in_dim``.
    labels : torch.Tensor, optional
        Tensor of outputs of the ``function`` with shape ``batch_size``.
    out_position : int, optional
        If the ``function`` is vector-valued, position of the output core in
        the resulting MPS.
    rank : int, optional
        Upper bound for the bond dimension of all cores.
    cum_percentage : float, optional
        When getting the proper bond dimension of each core via truncated SVD,
        this is the proportion that should be satisfied between the sum of all
        singular values kept and the total sum of all singular values. Therefore,
        it specifies the rank of each core independently, allowing for
        varying bond dimensions.
    batch_size : int
        Batch size used to process ``sketch_samples`` with ``DataLoaders``
        during the decomposition.
    device : torch.device, optional
        Device to which ``sketch_samples`` will be send to compute sketches. It
        should coincide with the device the ``function`` is in, in the case the
        function is a call to a ``nn.Module``.
    verbose : bool

    Returns
    -------
    list[torch.Tensor]
        List of tensor cores of the MPS.
    """
    if not isinstance(function, Callable):
        raise TypeError('`function` should be callable')
    
    if not isinstance(embedding, Callable):
        raise TypeError('`embedding` should be callable')
    
    # Number of input features
    if not isinstance(sketch_samples, torch.Tensor):
        raise TypeError('`sketch_samples` should be torch.Tensor type')
    if len(sketch_samples.shape) not in [2, 3]:
        # batch_size x n_features or batch_size x n_features x in_dim
        raise ValueError(
            '`sketch_samples` should be a tensor with shape (batch_size, '
            'n_features) or (batch_size, n_features, in_dim)')
    n_features = sketch_samples.size(1)
    if n_features == 0:
        raise ValueError('`sketch_samples` cannot be 0 dimensional')
    
    # Embedding dimension
    try:
        aux_embed = embedding(sketch_samples[:1, :1].to(device))
    except:
        raise ValueError(
            '`embedding` should take as argument a single tensor with shape '
            '(batch_size, n_features) or (batch_size, n_features, in_dim)')
        
    if len(aux_embed.shape) != 3:
        raise ValueError('`embedding` should return a tensor of shape '
                         '(batch_size, n_features, in_dim)')
    in_dim = aux_embed.size(2)
    if in_dim == 0:
        raise ValueError('Embedding dimension cannot be 0')
        
    # Output dimension
    try:
        aux_output = function(sketch_samples[:1].to(device))
    except:
        raise ValueError(
            '`function` should take as argument a single tensor with shape '
            '(batch_size, n_features) or (batch_size, n_features, in_dim)')
        
    if len(aux_output.shape) != 2:
        raise ValueError(
            '`function` should return a tensor of shape (batch_size, out_dim).'
            ' If `function` is scalar, out_dim = 1')
    out_dim = aux_output.size(1)
    if out_dim == 0:
        raise ValueError('Output dimension (of `function`) cannot be 0')
    
    # Output position
    if out_dim == 1:
        if out_position is not None:
            warnings.warn(
                '`out_position` will be ignored, since `function` is scalar')
        out_position = -1
    else:
        if out_position is None:
            out_position = (n_features + 1) // 2
        else:
            if not isinstance(out_position, int):
                raise TypeError('`out_position` should be int type')
            if (out_position  < 0) or (out_position > n_features):
                raise ValueError(
                    '`out_position` should be between 0 and the number of '
                    'features (equal to the second dimension of `sketch_samples`)'
                    ', both included')
    
    # Rank
    if rank is not None:
        if not isinstance(rank, int):
            raise TypeError('`rank` should be int type')
        if rank < 1:
            raise ValueError('`rank` should be greater or equal than 1')
    max_D = rank
    
    # Cum. percentage
    if cum_percentage is not None:
        if not isinstance(cum_percentage, float):
            raise TypeError('`cum_percentage` should be float type')
        if (cum_percentage <= 0) or (cum_percentage > 1):
            raise ValueError('`cum_percentage` should be in the range (0, 1]')
    
    # Batch size
    if not isinstance(batch_size, int):
        raise TypeError('`batch_size` should be int type')
    
    # Extend sketch_samples tensor with outputs
    if out_dim > 1:
        n_features += 1
        sketch_samples, labels = extend_with_output(function,
                                                    sketch_samples,
                                                    labels,
                                                    out_position,
                                                    batch_size,
                                                    device)
        
    # TODO: we are assuming data is continuous
    def aux_embedding(data):
        """
        For cases where ``n_features = 1``, it returns an embedded tensor with
        shape ``batch_size x out_dim``.
        """
        return embedding(data).squeeze(1)
    
    def aux_basis(data):
        """
        For cases where ``n_features = 1``, it returns an embedded tensor with
        shape ``batch_size x out_dim``.
        """
        return basis(data.int(), dim=out_dim).squeeze(1).float()
    
    cores = []
    D_k_1 = 1
    for k in range(n_features):
        # Prepare x_k
        if k == out_position:
            x_k = torch.arange(out_dim).view(-1, 1).float()
            phys_dim = out_dim
        else:
            # NOTE: this might be useful for 0 < x < 1, but not in general
            # NOTE: having x_k greater or equal than in_dim is ESSENTIAL
            x_k = torch.arange(in_dim).view(-1, 1).float() / in_dim
            
            # x_k = sketch_samples[:, k:(k + 1)].unique(dim=0)  # TODO: maybe sample from this to reduce amount
            # if x_k.size(0) > in_dim:
            #     perm = torch.randperm(x_k.size(0))
            #     idx = perm[:in_dim]
            #     x_k = x_k[idx]
            # elif x_k.size(0) < in_dim:
            #     perm = torch.randperm(x_k.size(0))
            #     idx = perm[:(in_dim - x_k.size(0))]
            #     x_k = torch.cat([x_k,
            #                      x_k[idx] + torch.randn_like(x_k[idx])],
            #                     dim=0)
            
            phys_dim = in_dim
        
        # Prepare T_k
        if k < (n_features - 1):
            T_k = sketch_samples[:, (k + 1):].unique(dim=0)
            
        if verbose:
            print(f'\n\n=========\nSite: {k}\n=========')
        D_k = min(D_k_1 * phys_dim, phys_dim ** (n_features - k - 1))
            
        if verbose:
            print(f'* Max D_k: min({D_k}, {max_D})')
            print(f'* T_k out dim: {T_k.size(0)}')
        D_k = min(D_k, max_D)
        
        if k == 0:
            # Sketching
            Phi_tilde_k = sketching(function, [x_k, T_k], out_position, batch_size, device)
            
            # Random unitary for T_k
            randu_t = random_unitary(Phi_tilde_k.size(1))#[:, :D_k]
            Phi_tilde_k = torch.mm(Phi_tilde_k, randu_t)
            
            if k != out_position:
                Phi_tilde_k = torch.linalg.lstsq(aux_embedding(x_k), Phi_tilde_k).solution
            
            # Trimming
            u, D_k = trimming(Phi_tilde_k, D_k, cum_percentage)
            B_k = u[:, :D_k]  # phys_dim x D_k
            
            # Solving
            cores.append(B_k)
            
            # Create S_k
            S_k = sketch_samples[:, :(k + 1)].unique(dim=0)
            s_k = S_k
            
            # Create A_k
            if k == out_position:
                aux_s_k = aux_basis(s_k)
            else:
                aux_s_k = aux_embedding(s_k)
            A_k = aux_s_k @ B_k
            
            # Set variables for next iteration
            D_k_1 = D_k
            A_k_1 = A_k
            S_k_1 = S_k
            
            if verbose:
                print(f'\nCore {k}:\n-------')
                print(cores[-1])
                print(f'* Final D_k: {D_k}')
                print(f'* S_k out dim: {S_k.size(0)}')
            
        elif k < (n_features - 1):
            # Sketching
            Phi_tilde_k = sketching(function, [S_k_1, x_k, T_k], out_position, batch_size, device)
            
            # Random unitary for T_k
            randu_t = random_unitary(Phi_tilde_k.size(2))\
                .repeat(Phi_tilde_k.size(0), 1, 1)#[..., :D_k]
            Phi_tilde_k = torch.bmm(Phi_tilde_k, randu_t)
            
            # # Random unitary for S_k_1
            # randu_s = random_unitary(Phi_tilde_k.size(0))\
            #     .repeat(Phi_tilde_k.size(2), 1, 1)[:, :D_k_1]
            # Phi_tilde_k = torch.bmm(randu_s, Phi_tilde_k.permute(2, 0, 1)).permute(1, 2, 0)
            
            if k != out_position:
                aux_Phi_tilde_k = torch.linalg.lstsq(
                    aux_embedding(x_k),
                    Phi_tilde_k.permute(1, 0, 2).reshape(x_k.size(0), -1)).solution
                
                Phi_tilde_k = aux_Phi_tilde_k.reshape(phys_dim,
                                                      Phi_tilde_k.size(0),
                                                      Phi_tilde_k.size(2)).permute(1, 0, 2)
            
            # Trimming
            u, D_k = trimming(Phi_tilde_k.reshape(-1, Phi_tilde_k.size(2)),
                              D_k,
                              cum_percentage)
            B_k = u[:, :D_k]  # (D_k_1 * phys_dim) x D_k
            
            # Solving
            G_k = torch.linalg.lstsq(A_k_1, B_k.reshape(-1, phys_dim * D_k)).solution
            G_k = G_k.view(-1, phys_dim, D_k)
            cores.append(G_k)
            
            # Create S_k
            S_k = sketch_samples[:, :(k + 1)].unique(dim=0)
            s_k = create_projector(S_k_1, S_k)
            
            # Create A_k
            A_k = B_k.view(-1, phys_dim, D_k)
            A_k = A_k[s_k[:, 0].long()]
            
            if k == out_position:
                aux_s_k = aux_basis(s_k[:, 1:])
            else:
                aux_s_k = aux_embedding(s_k[:, 1:])
            A_k = torch.einsum('bpd,bp->bd', A_k, aux_s_k)
            
            # Set variables for next iteration
            D_k_1 = D_k
            B_k_1 = B_k
            A_k_1 = A_k
            S_k_1 = S_k
            s_k_1 = s_k
            
            if verbose:
                print(f'\nCore {k}:\n-------')
                print(cores[-1])
                print(f'* Final D_k: {D_k}')
                print(f'* S_k out dim: {S_k.size(0)}')
            
        else:
            # Sketching
            Phi_tilde_k = sketching(function, [S_k_1, x_k], out_position, batch_size, device)
            
            # # Random unitary for S_k_1
            # randu_s = random_unitary(Phi_tilde_k.size(0))[:D_k_1, :]
            # Phi_tilde_k = torch.mm(randu_s, Phi_tilde_k)
            
            if k != out_position:
                Phi_tilde_k = torch.linalg.lstsq(aux_embedding(x_k), Phi_tilde_k.t()).solution.t()
                
            B_k = Phi_tilde_k
            
            # Solving
            G_k = torch.linalg.lstsq(A_k_1, B_k).solution
            cores.append(G_k)
            
            if verbose:
                print(f'\nCore {k}:\n-------')
                print(cores[-1])
            
    return cores
