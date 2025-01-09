"""
This script contains:

    * extend_with_output
    * sketching
    * trimming
    * create_projector
    * val_error
    * tt_rss
"""

import time
import warnings
from typing import Optional, Union, Callable, Tuple, List, Sequence, Text
from math import sqrt

import torch
from torch.utils.data import TensorDataset, DataLoader

from tensorkrowch.embeddings import basis
from tensorkrowch.utils import random_unitary
import tensorkrowch.models as models


def extend_with_output(function, samples, labels, out_position, batch_size, device):
    """
    Extends ``samples`` tensor with the output of the ``function`` evaluated on
    those ``samples``. If ``samples`` is a tensor with shape ``batch_size x
    n_features (x in_dim)``, the extended samples will have shape
    ``batch_size x (n_features + 1) (x in_dim)``.
    
    If ``labels`` are not provided, they are obtained by passing the ``samples``
    through the ``function``. In this case, it is assumed that ``function``
    returns a vector of squared roots of probabilities (for each class). Thus,
    the vector of ``labels`` is sampled according to the output distribution.
    
    If ``labels`` are given, it is assumed to be a tensor with shape ``batch_size``.
    """
    if labels is None:
        loader = DataLoader(TensorDataset(samples),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
        
        with torch.no_grad():
            outputs = []
            for (batch,) in loader:
                outputs.append(function(batch.to(device)).pow(2).cpu())
            
            outputs = torch.cat(outputs, dim=0).cpu()
            
            labels_distr = outputs.cumsum(dim=1)
            labels_distr = labels_distr / labels_distr[:, -1:]
            
            probs = torch.rand(outputs.size(0), 1)
            ids = outputs.size(1) - torch.le(probs,
                                             labels_distr).sum(dim=1, keepdim=True)
            outputs = outputs.gather(dim=1, index=ids).pow(0.5)
            
            # batch_size x n_features x in_dim
            if len(samples.shape) == 3:
                # In this case, labels are copied along dimension `in_dim`
                ids = ids.unsqueeze(2).expand(-1, -1, samples.shape[2])
            
            extended_samples = torch.cat([samples[:, :out_position],
                                          ids,
                                          samples[:, out_position:]], dim=1)
            return extended_samples, outputs
    else:
        ids = labels.view(-1, 1)
        outputs = torch.ones_like(ids).float()
        
        # batch_size x n_features x in_dim
        if len(samples.shape) == 3:
            # In this case, labels are the same along dimension `in_dim`
            ids = ids.unsqueeze(2).expand(-1, -1, samples.shape[2])
        
        extended_samples = torch.cat([samples[:, :out_position],
                                      ids,
                                      samples[:, out_position:]], dim=1)
        return extended_samples, outputs


def sketching(function, tensors_list, out_position, batch_size, device):
    """
    Given ``tensors_list``, a list of ``m`` tensors, where each tensor ``i`` has
    shape ``di x ni (x in_dim)`` and ``sum(n1, ..., nm) = n_features``, creates
    a projection tensor with shape ``d1 x ... x dm x n_features (x in_dim)``,
    which is passed to the ``function`` to compute ``Phi_tilde_k`` of shape
    ``d1 x ... x dm``.
    """
    sizes = []
    for tensor in tensors_list:
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.shape) in [2, 3]
        sizes.append(tensor.size(1))
    
    # Expand all tensors so that each one has shape d1 x ... x dm x ni
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
        
        if len(tensors_list[i].shape) == 3:
            # If shape is di x ni x in_dim, add in_dim to all tensors
            view_shape.append(tensors_list[i].size(2))
            expand_shape.append(-1)
        else:
            # If shape is di x ni, add extra aux dimension of 1
            view_shape.append(1)
            expand_shape.append(-1)
        
        aux_tensor = tensors_list[i].view(*view_shape).expand(*expand_shape)
        aux_list.append(aux_tensor)
    
    # Find labels in out_position
    if out_position >= 0: # If function is vector-valued
        cum_size = 0
        for i, size in enumerate(sizes):
            if (cum_size + size - 1) >= out_position:
                if size == 1:
                    labels = aux_list[i][..., 0, 0]
                    aux_list = aux_list[:i] + aux_list[(i + 1):]
                else:
                    labels = aux_list[i][..., out_position - cum_size, 0]
                    aux_list[i] = torch.cat(
                        [aux_list[i][..., :(out_position - cum_size), :],
                         aux_list[i][..., (out_position - cum_size + 1):, :]],
                        dim=-2)
                labels = labels.reshape(-1, 1).to(torch.int64).to(device)
                break
            cum_size += size
    else:
        labels = torch.zeros(*aux_list[0].shape[:-2], 1).to(torch.int64).to(device)
        labels = labels.view(-1, 1)
    
    projection = torch.cat(aux_list, dim=-2)
    
    if projection.shape[-1] == 1:
        projection_loader = DataLoader(
            TensorDataset(projection.view(-1, projection.shape[-2]),
                          labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
    else:
        # di x ni x in_dim
        projection_loader = DataLoader(
            TensorDataset(projection.view(-1,
                                          projection.shape[-2],
                                          projection.shape[-1]),
                          labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
    
    Phi_tilde_k = []
    with torch.no_grad():
        for batch, labs in projection_loader:
            aux_result = function(batch.to(device))
            Phi_tilde_k.append(aux_result.gather(dim=1,
                                                 index=labs).flatten().cpu())
    
    Phi_tilde_k = torch.cat(Phi_tilde_k, dim=0)   
    Phi_tilde_k = Phi_tilde_k.view(*projection.shape[:-2])
    
    return Phi_tilde_k


def trimming(mat, rank, cum_percentage):
    """Given a matrix returns the U from the SVD and an appropiate rank"""
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    
    if rank is None:
        rank = len(s)

    percentages = s.cumsum(0) / (s.sum().expand(s.shape) + 1e-10)
    cum_percentage_tensor = torch.tensor(cum_percentage)
    
    aux_rank = 0
    for p in percentages:
        if p == 0:
            if aux_rank == 0:
                aux_rank = 1
            break
        aux_rank += 1
        
        # Cut when ``cum_percentage`` is exceeded
        if p >= cum_percentage_tensor:
            break
        elif aux_rank >= rank:
            break
        
    return u, s, vh, aux_rank


def create_projector(S_k_1, S_k):
    """
    Given the previous projector and the current one, it infers the ``s_k``
    needed to create ``S_k`` from ``S_k_1``. All rows of ``S_k_1`` and ``S_k``
    must be unique.

    Parameters
    ----------
    S_k_1 : torch.Tensor
        Matrix of shape ``n x k (x in_dim)``. The ``n`` rows of ``S_k_1`` are
        equal to the rows of ``S_k[:, :-1]``, but maybe they are repeated in
        ``S_k``.
    S_k : torch.Tensor
        Matrix of shape ``m x (k + 1) (x in_dim)``, with m >= n.
    
    Returns
    -------
    s_k : torch.Tensor
        Tensor of shape ``m x 2 (x in_dim)``. The 2 columns correspond,
        respectively, to indices of rows of ``S_k_1`` (index 0) and the new
        elements in the ``(k + 1)``-th column of ``S_k`` (index 1) associated
        to the corresponding rows of ``S_k_1``.
    
    Example
    -------
    >>> S_k = torch.randint(low=0, high=2, size=(10, 4)).unique(dim=0)
    >>> S_k
    
    tensor([[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0]])
            
    >>> S_k_1 = S_k[:, :-1].unique(dim=0)
    >>> S_k_1
    
    tensor([[0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]])
            
    >>> create_projector(S_k_1, S_k)
    
    [tensor([0, 1, 2, 3, 4, 5, 6]),
    tensor([[1],
            [0],
            [1],
            [1],
            [0],
            [0],
            [0]])]
    """
    if len(S_k.shape) == 2:
        # n x k
        s_k_0 = torch.empty_like(S_k[:, -1]).long()
        where_equal_dim = 1
    else:
        # n x k x in_dim
        s_k_0 = torch.empty_like(S_k[:, -1, -1]).long()
        where_equal_dim = (1, 2)
    s_k_1 = torch.empty_like(S_k[:, -1:])
        
    for i in range(S_k_1.size(0)):
        where_equal = (S_k[:, :-1] == S_k_1[i]).all(dim=where_equal_dim)
        new_col = S_k[where_equal, -1:]
        first_col = torch.Tensor([i]).expand(new_col.size(0)).to(new_col.device)
        
        s_k_0[where_equal] = first_col.long()
        s_k_1[where_equal] = new_col
        
    s_k = [s_k_0, s_k_1]
    return s_k  


@torch.no_grad()
def val_error(function, embedding, cores, sketch_samples, out_position, device):
    """Computes relative error on ``sketch_samples``."""
    if out_position > -1:
        sketch_samples = torch.cat([sketch_samples[:, :out_position],
                                    sketch_samples[:, (out_position + 1):]],
                                   dim=1)
    
    exact_output = function(sketch_samples.to(device))
    
    if exact_output.size(1) > 1:
        mps = models.MPSLayer(tensors=[c.to(device) for c in cores])
    else:
        mps = models.MPS(tensors=[c.to(device) for c in cores])
    
    embed_samples = embedding(sketch_samples.to(device))                       
    approx_output = mps(embed_samples, inline_input=True, inline_mats=True)
    
    if exact_output.size(1) == 1:
        exact_output = exact_output.squeeze(1)
    
    # TODO: add small epsilon in denominator to avoid dividing by 0
    error = (exact_output - approx_output).norm() / exact_output.norm()
    return error


# MARK: TT-RSS
@torch.no_grad()
def tt_rss(function: Callable,
           embedding: Callable,
           sketch_samples: torch.Tensor,
           labels: Optional[torch.Tensor] = None,
           domain: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
           domain_multiplier: int = 1,
           out_position: Optional[int] = None,
           rank: Optional[int] = None,
           cum_percentage: Optional[float] = None,
           batch_size: int = 64,
           device: Optional[torch.device] = None,
           verbose: bool = True,
           return_info: bool = False) -> Union[List[torch.Tensor],
                                               Tuple[List[torch.Tensor], dict]]:
    r"""
    Tensor Train via Recursive Sketching from Samples.
    
    Decomposes a scalar or vector-valued function of :math:`N` input variables
    in a Matrix Product State of :math:`N` cores, each corresponding to one
    input variable, in the same order as they are provided to the function. To
    turn each input variable into a vector that can be contracted with the
    corresponding MPS core, an embedding function is required. The dimension of
    the embedding will be used as the input dimension of the MPS.
    
    If the function is vector-valued, it will be seen as a :math:`N + 1` scalar
    function, returning a MPS with :math:`N + 1` cores. The output variable will
    use the embedding :func:`~tensorkrowch.basis`, which maps integers
    (corresponding to indices of the output vector) to basis vectors:
    :math:`i \mapsto \langle i \rvert`. It can be specified the position in
    which the output core will be. By default, it will be in the middle of the
    MPS.
    
    To specify the bond dimension of each MPS core, one can use the arguments
    ``rank`` and ``cum_percentage``. If more than one is specified, the
    resulting rank will be the one that satisfies all conditions.
    
    Parameters
    ----------
    function : Callable
        Function that is going to be decomposed. It needs to have a single
        input argument, the data, which is a tensor of shape
        ``batch_size x n_features`` or ``batch_size x n_features x in_dim``. It
        must return a tensor of shape ``batch_size x out_dim``. If the function
        is scalar, ``out_dim = 1``.
    embedding : Callable
        Embedding function that maps the data tensor to a higher dimensional
        space. It needs to have a single argument. It is a function that
        transforms the given data tensor of shape ``batch_size x n_features`` or
        ``batch_size x n_features x in_dim`` and returns an embedded tensor of
        shape ``batch_size x n_features x embed_dim``.
    sketch_samples : torch.Tensor
        Samples that will be used as sketches to decompose the function. It has
        to be a tensor of shape ``batch_size x n_features`` or
        ``batch_size x n_features x in_dim``.
    labels : torch.Tensor, optional
        Tensor of output labels of the ``function`` with shape ``batch_size``.
        If ``function`` is vector-valued, ``labels`` will be used to select
        an element from each output vector. If ``labels`` are not given, these
        will be obtained according to the distribution represented by the output
        vectors (assuming these represent square roots of probabilities for each
        class).
    domain : torch.Tensor or list[torch.Tensor], optional
        Domain of the input variables. It should be given as a finite set of
        possible values that can take each variable. If all variables live in
        the same domain, it should be given as a tensor with shape ``n_values``
        or ``n_values x in_dim``, where the possible ``n_values`` should be at
        least as large as the desired input dimension of the MPS cores, which
        is the ``embed_dim`` of the ``embedding``. The more values are given,
        the more accurate will be the tensorization but more costly will be to
        do it. If ``domain`` is given as a list, it should have the same
        number of elements as input variables, so that each variable can live
        in a different domain. If ``domain`` is not given, it will be obtained
        from the values each variable takes in the ``sketch_samples``.
    domain_multiplier : int
        Upper bound for how many values are used for the input variable domain
        if ``domain`` is not provided. If ``domain`` is not provided, the
        domain of the input variables will be inferred from the unique values
        each variable takes in the ``sketch_samples``. In this case, only
        ``domain_multiplier * embed_dim`` values will be taken randomly.
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
        Device to which ``sketch_samples`` will be sent to compute sketches. It
        should coincide with the device the ``function`` is in, in the case the
        function is a call to a ``nn.Module`` or uses tensors that are in a 
        specific device. This also applies to the ``embedding`` function.
    verbose : bool
        Default is ``True``.
    return_info : bool
        Boolean indicating if an additional dictionary with total time and
        validation error should be returned.

    Returns
    -------
    list[torch.Tensor]
        List of tensor cores of the MPS.
    dictionary
        If ``return_info`` is ``True``.
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
                         '(batch_size, n_features, embed_dim)')
    embed_dim = aux_embed.size(2)
    if embed_dim == 0:
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
    
    # Labels
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError('`labels` should be torch.Tensor type')
        if labels.shape != sketch_samples.shape[:1]:
            raise ValueError('`labels` should be a tensor with shape (batch_size,)')
    
    # Input domain
    if domain is not None:
        if not isinstance(domain, torch.Tensor):
            if not isinstance(domain, Sequence):
                raise TypeError(
                    '`domain` should be torch.Tensor or list[torch.Tensor] type')
            else:
                for t in domain:
                    if not isinstance(t, torch.Tensor):
                        raise TypeError(
                            '`domain` should be torch.Tensor or list[torch.Tensor]'
                            ' type')
                
                if len(domain) != n_features:
                    raise ValueError(
                        'If `domain` is given as a sequence of tensors, it should'
                        ' have as many elements as input variables')
        else:
            if len(domain.shape) != (len(sketch_samples.shape) - 1):
                raise ValueError(
                    'If `domain` is given as a torch.Tensor, it should have '
                    'shape (n_values,) or (n_values, in_dim), and it should '
                    'only include in_dim if it also appears in the shape of '
                    '`sketch_samples`')
            if len(domain.shape) == 2:
                if domain.shape[1] == 1:
                    raise ValueError()
    
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
    
    # Cum. percentage
    if cum_percentage is not None:
        if not isinstance(cum_percentage, float):
            raise TypeError('`cum_percentage` should be float type')
        if (cum_percentage <= 0) or (cum_percentage > 1):
            raise ValueError('`cum_percentage` should be in the range (0, 1]')
    
    if (rank is None) and (cum_percentage is None):
        raise ValueError(
            'At least one of `rank` and `cum_percentage` should be given')
    
    # Batch size
    if not isinstance(batch_size, int):
        raise TypeError('`batch_size` should be int type')
    
    # Extend sketch_samples tensor with outputs
    if out_dim > 1:
        n_features += 1
        sketch_samples, _ = extend_with_output(function=function,
                                               samples=sketch_samples,
                                               labels=labels,
                                               out_position=out_position,
                                               batch_size=batch_size,
                                               device=device)
        
    def aux_embedding(data):
        """
        For the cases where ``n_features = 1``, it returns an embedded tensor
        with shape ``batch_size x embed_dim``.
        """
        return embedding(data).squeeze(1)
    
    def aux_basis(data):
        """
        For the cases where ``n_features = 1``, it returns an embedded tensor
        with shape ``batch_size x basis_dim``.
        """
        # batch_size x n_features(=1) x in_dim
        if len(data.shape) == 3:
            # In this case, labels are the same along dimension `in_dim`
            data = data[:, :, 0]
        return basis(data.int(), dim=out_dim).squeeze(1).float()
    
    start_time = time.time()
    cores = []
    D_k_1 = 1
    for k in range(n_features):
        
        # Prepare x_k
        if k == out_position:
            x_k = torch.arange(out_dim).view(-1, 1).float()
            phys_dim = out_dim
        
        else:
            if domain is not None:
                if isinstance(domain, torch.Tensor):
                    x_k = domain.unsqueeze(1)
                else:
                    x_k = domain[k if k < out_position else (k - 1)].unsqueeze(1)
            
            else:
                x_k = sketch_samples[:, k:(k + 1)].unique(dim=0)
                
                if x_k.size(0) >= (domain_multiplier * embed_dim):
                    perm = torch.randperm(x_k.size(0))
                    idx = perm[:(domain_multiplier * embed_dim)]
                    x_k = x_k[idx]
            
            phys_dim = embed_dim
        
        # Prepare T_k
        if k < (n_features - 1):
            T_k = sketch_samples[:, (k + 1):].unique(dim=0)
        
        # Prepare D_k
        if verbose:
            site_count = f'|| Site: {k + 1} / {n_features} ||'
            site_count = ['=' * len(site_count), site_count]
            print('\n\n' + site_count[0] + '\n' + site_count[1] + '\n' + site_count[0])
        
        D_k = min(D_k_1 * phys_dim, phys_dim ** (n_features - k - 1))
        
        if verbose:
            if rank is None:
                print(f'* Max D_k: {D_k}')
            else:
                print(f'* Max D_k: min({D_k}, {rank})')
            print(f'* T_k out dim: {T_k.size(0)}')
        
        if rank is not None:
            D_k = min(D_k, rank)
        
        # Tensorize
        if k == 0:
            # Sketching
            Phi_tilde_k = sketching(function=function,
                                    tensors_list=[x_k, T_k],
                                    out_position=out_position,
                                    batch_size=batch_size,
                                    device=device)
            
            # Random unitary for T_k
            randu_t = random_unitary(Phi_tilde_k.size(1)).to(Phi_tilde_k.dtype)
            Phi_tilde_k = torch.mm(Phi_tilde_k, randu_t)
            
            if k != out_position:
                Phi_tilde_k = torch.linalg.lstsq(
                    aux_embedding(x_k.to(device)).cpu(),
                    Phi_tilde_k).solution
            
            # Trimming
            u, _, _, D_k = trimming(mat=Phi_tilde_k,
                                    rank=D_k,
                                    cum_percentage=cum_percentage)
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
                aux_s_k = aux_embedding(s_k.to(device)).cpu()
            A_k = aux_s_k @ B_k
            
            # Set variables for next iteration
            D_k_1 = D_k
            A_k_1 = A_k
            S_k_1 = S_k
            
            if verbose:
                core_count = f'Core {k + 1}:'
                print('\n' + core_count  + '\n' + ('-' * len(core_count)))
                print(cores[-1])
                print(f'* Final D_k: {D_k}')
                print(f'* S_k out dim: {S_k.size(0)}')
            
        elif k < (n_features - 1):
            # Sketching
            Phi_tilde_k = sketching(function=function,
                                    tensors_list=[S_k_1, x_k, T_k],
                                    out_position=out_position,
                                    batch_size=batch_size,
                                    device=device)
            
            # Random unitary for T_k
            randu_t = random_unitary(Phi_tilde_k.size(2))\
                .repeat(Phi_tilde_k.size(0), 1, 1).to(Phi_tilde_k.dtype)
            Phi_tilde_k = torch.bmm(Phi_tilde_k, randu_t)
            
            if k != out_position:
                aux_Phi_tilde_k = torch.linalg.lstsq(
                    aux_embedding(x_k.to(device)).cpu(),
                    Phi_tilde_k.permute(1, 0, 2).reshape(x_k.size(0), -1)).solution
                
                Phi_tilde_k = aux_Phi_tilde_k.reshape(
                    phys_dim,
                    Phi_tilde_k.size(0),
                    Phi_tilde_k.size(2)).permute(1, 0, 2)
            
            # Trimming
            u, _, _, D_k = trimming(mat=Phi_tilde_k.reshape(-1,
                                                            Phi_tilde_k.size(2)),
                                    rank=D_k,
                                    cum_percentage=cum_percentage)
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
            A_k = A_k[s_k[0]]
            
            if k == out_position:
                aux_s_k = aux_basis(s_k[1])
            else:
                aux_s_k = aux_embedding(s_k[1].to(device)).cpu()
            A_k = torch.einsum('bpd,bp->bd', A_k, aux_s_k)
            
            # Set variables for next iteration
            D_k_1 = D_k
            A_k_1 = A_k
            S_k_1 = S_k
            
            if verbose:
                core_count = f'Core {k + 1}:'
                print('\n' + core_count  + '\n' + ('-' * len(core_count)))
                print(cores[-1])
                print(f'* Final D_k: {D_k}')
                print(f'* S_k out dim: {S_k.size(0)}')
            
        else:
            # Sketching
            Phi_tilde_k = sketching(function=function,
                                    tensors_list=[S_k_1, x_k],
                                    out_position=out_position,
                                    batch_size=batch_size,
                                    device=device)
            
            if k != out_position:
                Phi_tilde_k = torch.linalg.lstsq(
                    aux_embedding(x_k.to(device)).cpu(),
                    Phi_tilde_k.t()).solution.t()
            
            # Trimming
            B_k = Phi_tilde_k
            
            # Solving
            G_k = torch.linalg.lstsq(A_k_1, B_k).solution
            cores.append(G_k)
            
            if verbose:
                core_count = f'Core {k + 1}:'
                print('\n' + core_count  + '\n' + ('-' * len(core_count)))
                print(cores[-1])
    
    if return_info:
        total_time = time.time() - start_time
        error = val_error(function=function,
                          embedding=embedding,
                          cores=cores,
                          sketch_samples=sketch_samples,
                          out_position=out_position,
                          device=device)
    
        info = {'total_time': total_time,
                'val_eps': error}
        
        return cores, info
            
    return cores
