"""
Tests for mps_layer:

    * TestMPSLayer
    * TestUMPSLayer
    * TestConvMPSLayer
    * TestConvUMPSLayer
"""

import pytest

import torch
import tensorkrowch as tk


class TestMPSLayer:

    def test_all_algorithms(self):
        example = torch.randn(1, 4, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 4, 5)

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.MPSLayer(n_features=4,
                                         in_dim=5,
                                         out_dim=12,
                                         bond_dim=2,
                                         out_position=out_position,
                                         boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1

                                # Canonicalize and continue
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(mode=mode, **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100, 12)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 2
                                        else:
                                            assert len(mps.virtual_nodes) == 1

    def test_all_algorithms_diff_in_dim(self):
        in_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.MPSLayer(n_features=4,
                                         in_dim=in_dim,
                                         out_dim=12,
                                         bond_dim=2,
                                         out_position=out_position,
                                         boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0

                                # Canonicalize and continue
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(mode=mode, **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100, 12)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0

    def test_all_algorithms_diff_bond_dim(self):
        bond_dim = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = torch.randn(1, 4, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 4, 5)

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.MPSLayer(
                    n_features=4,
                    in_dim=5,
                    out_dim=12,
                    bond_dim=bond_dim[:-1] if boundary == 'obc' else bond_dim,
                    out_position=out_position,
                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1

                                # Canonicalize and continue
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(mode=mode, **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100, 12)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 2
                                        else:
                                            assert len(mps.virtual_nodes) == 1

    def test_all_algorithms_diff_in_dim_bond_dim(self):
        in_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        bond_dim = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.MPSLayer(
                    n_features=4,
                    in_dim=in_dim,
                    out_dim=12,
                    bond_dim=bond_dim[:-1] if boundary == 'obc' else bond_dim,
                    out_position=out_position,
                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0

                                # Canonicalize and continue
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(mode=mode, **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100, 12)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0

    def test_extreme_case_left_output(self):
        # Left node + Outpt node
        mps = tk.models.MPSLayer(n_features=1,
                                 in_dim=5,
                                 out_dim=12,
                                 bond_dim=2,
                                 out_position=1,
                                 boundary='obc')
        example = torch.randn(1, 1, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 1, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 1

                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 2
                                assert len(mps.data_nodes) == 1
                                assert len(mps.virtual_nodes) == 1

    def test_extreme_case_output_right(self):
        # Output node + Right node
        mps = tk.models.MPSLayer(n_features=1,
                                 in_dim=5,
                                 out_dim=12,
                                 bond_dim=2,
                                 out_position=0,
                                 boundary='obc')
        example = torch.randn(1, 1, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 1, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 1

                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 2
                                assert len(mps.data_nodes) == 1
                                assert len(mps.virtual_nodes) == 1

    def test_extreme_case_output(self):
        # Outpt node
        mps = tk.models.MPSLayer(n_features=0,
                                 in_dim=5,
                                 out_dim=12,
                                 bond_dim=2,
                                 out_position=0,
                                 boundary='pbc')

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace(inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(inline_input=inline_input,
                                     inline_mats=inline_mats)  # Same as mps.contract()

                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 0

    def test_canonicalize_univocal(self):
        example = torch.randn(1, 4, 2)  # batch x n_features x feature_dim
        data = torch.randn(100, 4, 2)

        mps = tk.models.MPSLayer(n_features=4,
                                 in_dim=2,
                                 out_dim=2,
                                 bond_dim=5,
                                 boundary='obc')

        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor

        # mps.output_node.tensor = torch.randn(mps.output_node.shape)

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100, 2)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 1

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-4

    def test_canonicalize_univocal_diff_dims(self):
        in_dim = [2, 3, 5, 6]  # torch.arange(2, 6).int().tolist()
        bond_dim = torch.arange(2, 6).int().tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        mps = tk.models.MPSLayer(n_features=4,
                                 in_dim=in_dim,
                                 out_dim=5,
                                 bond_dim=bond_dim,
                                 boundary='obc')

        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor

        # mps.output_node.tensor = torch.randn(mps.output_node.shape)

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100, 5)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 0

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5

    def test_canonicalize_univocal_bond_greater_than_phys(self):
        example = torch.randn(1, 4, 2)  # batch x n_features x feature_dim
        data = torch.randn(100, 4, 2)

        mps = tk.models.MPSLayer(n_features=4,
                                 in_dim=2,
                                 out_dim=2,
                                 bond_dim=20,
                                 boundary='obc')

        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor

        # mps.output_node.tensor = torch.randn(mps.output_node.shape)

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100, 2)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 1

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5

    def test_check_grad(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mps = tk.models.MPSLayer(n_features=1,
                                 in_dim=2,
                                 out_dim=2,
                                 bond_dim=2,
                                 out_position=1,
                                 boundary='obc').to(device)

        data = torch.randn(1, 1, 2).to(device)
        result = mps.forward(data)
        result[0, 0].backward()

        I = data.squeeze(0)
        A = mps.left_node.tensor
        B = mps.output_node.tensor
        grad_A1 = mps.left_node.grad
        grad_B1 = mps.output_node.grad

        grad_A2 = I.t() @ B[:, 0].view(2, 1).t()
        grad_B2 = (I @ A).t() @ torch.tensor([[1., 0.]]).to(device)

        assert torch.equal(grad_A1, grad_A2)
        assert torch.equal(grad_B1, grad_B2)


class TestUMPSLayer:

    def test_all_algorithms(self):
        example = torch.randn(1, 4, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 4, 5)

        for out_position in range(5):
            mps = tk.models.UMPSLayer(n_features=4,
                                      in_dim=5,
                                      out_dim=12,
                                      bond_dim=2,
                                      out_position=out_position)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100, 12)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 5
                            assert len(mps.data_nodes) == 4
                            assert len(mps.virtual_nodes) == 2

    def test_extreme_case_left_output(self):
        # Left node + Output node
        mps = tk.models.UMPSLayer(n_features=1,
                                  in_dim=5,
                                  out_dim=12,
                                  bond_dim=2,
                                  out_position=1)
        example = torch.randn(1, 1, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 1, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 2

    def test_extreme_case_output_right(self):
        # Output node + Right node
        mps = tk.models.UMPSLayer(n_features=1,
                                  in_dim=5,
                                  out_dim=12,
                                  bond_dim=2,
                                  out_position=0)
        example = torch.randn(1, 1, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 1, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 2

    def test_extreme_case_output(self):
        # Outpt node
        mps = tk.models.UMPSLayer(n_features=0,
                                  in_dim=5,
                                  out_dim=12,
                                  bond_dim=2,
                                  out_position=0)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace(inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(inline_input=inline_input,
                                     inline_mats=inline_mats)  # Same as mps.contract()

                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 1


class TestConvMPSLayer:

    def test_all_algorithms(self):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.ConvMPSLayer(in_channels=2,
                                             out_channels=5,
                                             bond_dim=2,
                                             kernel_size=2,
                                             out_position=out_position,
                                             boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for conv_mode in ['flat', 'snake']:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)

                                    assert result.shape == (100, 5, 4, 4)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1

                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(
                                                mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats,
                                                      mode=conv_mode)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats,
                                                         mode=conv_mode)

                                            assert result.shape == (
                                                100, 5, 4, 4)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and auto_stack:
                                                assert len(
                                                    mps.virtual_nodes) == 2
                                            else:
                                                assert len(
                                                    mps.virtual_nodes) == 1

    def test_all_algorithms_diff_bond_dim(self):
        bond_dim = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        for boundary in ['obc', 'pbc']:
            for out_position in range(5):
                mps = tk.models.ConvMPSLayer(
                    in_channels=2,
                    out_channels=5,
                    bond_dim=bond_dim[:-1] if boundary == 'obc' else bond_dim,
                    kernel_size=2,
                    out_position=out_position,
                    boundary=boundary)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for conv_mode in ['flat', 'snake']:
                                    mps.auto_stack = auto_stack
                                    mps.auto_unbind = auto_unbind

                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)

                                    assert result.shape == (100, 5, 4, 4)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1

                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(
                                                mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats,
                                                      mode=conv_mode)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats,
                                                         mode=conv_mode)

                                            assert result.shape == (
                                                100, 5, 4, 4)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and auto_stack:
                                                assert len(
                                                    mps.virtual_nodes) == 2
                                            else:
                                                assert len(
                                                    mps.virtual_nodes) == 1

    def test_extreme_case_left_output(self):
        # Left node + Outpt node
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        mps = tk.models.ConvMPSLayer(in_channels=2,
                                     out_channels=5,
                                     bond_dim=2,
                                     kernel_size=1,
                                     out_position=1,
                                     boundary='obc')

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)

                                    assert result.shape == (100, 5, 5, 5)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 1
                                    assert len(mps.virtual_nodes) == 1

    def test_extreme_case_output_right(self):
        # Output node + Right node
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        mps = tk.models.ConvMPSLayer(in_channels=2,
                                     out_channels=5,
                                     bond_dim=2,
                                     kernel_size=1,
                                     out_position=0,
                                     boundary='obc')

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)

                                    assert result.shape == (100, 5, 5, 5)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 1
                                    assert len(mps.virtual_nodes) == 1


class TestConvUMPSLayer:

    def test_all_algorithms(self):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        for out_position in range(5):
            mps = tk.models.ConvUMPSLayer(in_channels=2,
                                          out_channels=5,
                                          bond_dim=2,
                                          kernel_size=2,
                                          out_position=out_position)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            for conv_mode in ['flat', 'snake']:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          mode=conv_mode)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             mode=conv_mode)

                                assert result.shape == (100, 5, 4, 4)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                assert len(mps.virtual_nodes) == 2

    def test_extreme_case_left_output(self):
        # Left node + Output node
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        mps = tk.models.ConvUMPSLayer(in_channels=2,
                                      bond_dim=2,
                                      out_channels=5,
                                      kernel_size=1,
                                      out_position=1)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 2

    def test_extreme_case_output_right(self):
        # Output node + Right node
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        mps = tk.models.ConvUMPSLayer(in_channels=2,
                                      bond_dim=2,
                                      out_channels=5,
                                      kernel_size=1,
                                      out_position=0)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 2
