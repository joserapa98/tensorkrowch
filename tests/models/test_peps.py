"""
Tests for peps:

    * TestPEPS
    * TestUPEPS
    * TestConvPEPS
    * TestConvUPEPS
"""

import pytest

import torch
import tensorkrowch as tk


class TestPEPS:

    def test_all_algorithms(self):
        example = torch.randn(1, 3*4, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 3*4, 5)

        for boundary_0 in ['obc', 'pbc']:
            for boundary_1 in ['obc', 'pbc']:
                peps = tk.models.PEPS(n_rows=3,
                               n_cols=4,
                               in_dim=5,
                               bond_dim=[2, 3],
                               boundary=[boundary_0, boundary_1],)

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for side in ['up', 'down', 'left', 'right']:
                            for inline in [True, False]:
                                peps.auto_stack = auto_stack
                                peps.auto_unbind = auto_unbind

                                peps.trace(example,
                                           from_side=side,
                                           inline=inline)
                                result = peps(data,
                                              from_side=side,
                                              inline=inline)

                                assert result.shape == (100,)
                                assert len(peps.edges) == 0
                                assert len(peps.leaf_nodes) == 3*4
                                assert len(peps.data_nodes) == 3*4
                                if auto_stack:
                                    if boundary_0 == 'obc':
                                        if boundary_1 == 'obc':
                                            assert len(peps.virtual_nodes) == 6
                                        else:
                                            assert len(peps.virtual_nodes) == 4
                                    else:
                                        if boundary_1 == 'obc':
                                            assert len(peps.virtual_nodes) == 4
                                        else:
                                            assert len(peps.virtual_nodes) == 2
                                else:
                                    assert len(peps.virtual_nodes) == 1

    def test_extreme_case_2_rows_2_cols(self):
        n_rows = 2
        n_cols = 2
        boundary = ['obc', 'obc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_3_cols(self):
        n_rows = 1
        n_cols = 3
        boundary = ['pbc', 'obc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 4
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_2_cols(self):
        n_rows = 1
        n_cols = 2
        boundary = ['pbc', 'obc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 3
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_3_rows_1_col(self):
        n_rows = 3
        n_cols = 1
        boundary = ['obc', 'pbc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 4
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_2_rows_1_col(self):
        n_rows = 2
        n_cols = 1
        boundary = ['obc', 'pbc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 3
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_1_col(self):
        n_rows = 1
        n_cols = 1
        boundary = ['pbc', 'pbc']

        peps = tk.models.PEPS(n_rows=n_rows,
                       n_cols=n_cols,
                       in_dim=5,
                       bond_dim=[2, 3],
                       boundary=boundary)
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 2
                        else:
                            assert len(peps.virtual_nodes) == 1


class TestUPEPS:

    def test_all_algorithms(self):
        example = torch.randn(1, 3*4, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, 3*4, 5)

        peps = tk.models.UPEPS(n_rows=3,
                        n_cols=4,
                        in_dim=5,
                        bond_dim=[2, 3])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == 3*4
                        assert len(peps.data_nodes) == 3*4
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_2_rows_2_cols(self):
        n_rows = 2
        n_cols = 2

        peps = tk.models.UPEPS(n_rows=n_rows,
                        n_cols=n_cols,
                        in_dim=5,
                        bond_dim=[2, 3])
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_1_row_2_cols(self):
        n_rows = 1
        n_cols = 2

        peps = tk.models.UPEPS(n_rows=n_rows,
                        n_cols=n_cols,
                        in_dim=5,
                        bond_dim=[2, 3])
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_2_rows_1_col(self):
        n_rows = 2
        n_cols = 1

        peps = tk.models.UPEPS(n_rows=n_rows,
                        n_cols=n_cols,
                        in_dim=5,
                        bond_dim=[2, 3])
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_1_row_1_col(self):
        n_rows = 1
        n_cols = 1

        peps = tk.models.UPEPS(n_rows=n_rows,
                        n_cols=n_cols,
                        in_dim=5,
                        bond_dim=[2, 3])
        example = torch.randn(1, n_rows * n_cols, 5)  # batch x n_features x feature_dim
        data = torch.randn(100, n_rows * n_cols, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100,)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2


class TestConvPEPS:

    def test_all_algorithms(self):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        for boundary_0 in ['obc', 'pbc']:
            for boundary_1 in ['obc', 'pbc']:
                peps = tk.models.ConvPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=3,
                                   boundary=[boundary_0, boundary_1])

                for auto_stack in [True, False]:
                    for auto_unbind in [True, False]:
                        for side in ['up', 'down', 'left', 'right']:
                            for inline in [True, False]:
                                peps.auto_stack = auto_stack
                                peps.auto_unbind = auto_unbind

                                peps.trace(example,
                                           from_side=side,
                                           inline=inline,
                                           max_bond=8)
                                result = peps(data,
                                              from_side=side,
                                              inline=inline,
                                              max_bond=8)

                                assert result.shape == (100, 3, 3)
                                assert len(peps.edges) == 0
                                assert len(peps.leaf_nodes) == 3*3
                                assert len(peps.data_nodes) == 3*3
                                if auto_stack:
                                    if boundary_0 == 'obc':
                                        if boundary_1 == 'obc':
                                            assert len(peps.virtual_nodes) == 6
                                        else:
                                            assert len(peps.virtual_nodes) == 4
                                    else:
                                        if boundary_1 == 'obc':
                                            assert len(peps.virtual_nodes) == 4
                                        else:
                                            assert len(peps.virtual_nodes) == 2
                                else:
                                    assert len(peps.virtual_nodes) == 1

    def test_extreme_case_2_rows_2_cols(self):
        n_rows = 2
        n_cols = 2

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['obc', 'obc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 4, 4)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_3_cols(self):
        n_rows = 1
        n_cols = 3

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['pbc', 'obc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 5, 3)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 4
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_2_cols(self):
        n_rows = 1
        n_cols = 2

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['pbc', 'obc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 5, 4)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 3
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_3_rows_1_col(self):
        n_rows = 3
        n_cols = 1

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['obc', 'pbc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 3, 5)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 4
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_2_rows_1_col(self):
        n_rows = 2
        n_cols = 1

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['obc', 'pbc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 4, 5)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 3
                        else:
                            assert len(peps.virtual_nodes) == 1

    def test_extreme_case_1_row_1_col(self):
        n_rows = 1
        n_cols = 1

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                           bond_dim=[2, 3],
                           kernel_size=(n_rows, n_cols),
                           boundary=['pbc', 'pbc'])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 5, 5)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        if auto_stack:
                            assert len(peps.virtual_nodes) == 2
                        else:
                            assert len(peps.virtual_nodes) == 1


class TestConvUPEPS:

    def test_all_algorithms(self):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                            bond_dim=[2, 3],
                            kernel_size=3)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline,
                                   max_bond=8)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline,
                                      max_bond=8)

                        assert result.shape == (100, 3, 3)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == 3*3
                        assert len(peps.data_nodes) == 3*3
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_2_rows_2_cols(self):
        n_rows = 2
        n_cols = 2

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                            bond_dim=[2, 3],
                            kernel_size=(n_rows, n_cols))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 4, 4)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_1_row_2_cols(self):
        n_rows = 1
        n_cols = 2

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                            bond_dim=[2, 3],
                            kernel_size=(n_rows, n_cols))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 5, 4)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_2_rows_1_col(self):
        n_rows = 2
        n_cols = 1

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                            bond_dim=[2, 3],
                            kernel_size=(n_rows, n_cols))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 4, 5)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2

    def test_extreme_case_1_row_1_col(self):
        n_rows = 1
        n_cols = 1

        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                            bond_dim=[2, 3],
                            kernel_size=(n_rows, n_cols))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for side in ['up', 'down', 'left', 'right']:
                    for inline in [True, False]:
                        print(auto_stack, auto_unbind, side, inline)
                        peps.auto_stack = auto_stack
                        peps.auto_unbind = auto_unbind

                        peps.trace(example,
                                   from_side=side,
                                   inline=inline)
                        result = peps(data,
                                      from_side=side,
                                      inline=inline)

                        assert result.shape == (100, 5, 5)
                        assert len(peps.edges) == 0
                        assert len(peps.leaf_nodes) == n_rows * n_cols
                        assert len(peps.data_nodes) == n_rows * n_cols
                        assert len(peps.virtual_nodes) == 2
