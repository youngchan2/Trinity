"""
Tensor shape definitions for Trinity benchmarks.

This module provides centralized shape definitions for forward and backward passes.
"""

from typing import Dict, Tuple, Union


class TensorShapeBuilder:
    """Builder class for tensor shapes used in Trinity benchmarks."""

    def __init__(self, M: int, N: int, D: int, H: int, P: int):
        """Initialize with dimension parameters.

        Args:
            M: Batch size
            N: Feature dimension
            D: Head dimension
            H: Number of heads
            P: Cache length
        """
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

    def get_forward_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get tensor shapes for forward pass.

        Returns:
            Dictionary mapping tensor names to their shapes (tuples of ints)
        """
        return {
            # Input and normalized tensors
            'X': (self.M, self.N),
            'X2': (self.M, ),
            'X_norm': (self.M, self.N),

            # Weight matrices
            'WQ': (self.N, self.N),
            'WK': (self.N, self.N),
            'WV': (self.N, self.N),

            # Query, Key, Value tensors (3D)
            'Q': (self.H, self.M, self.D),
            'K': (self.H, self.M, self.D),
            'V': (self.H, self.M, self.D),

            # Query, Key, Value tensors (2D)
            'Q1': (self.M, self.N),
            'K1': (self.M, self.N),
            'V1': (self.M, self.N),

            # Query, Key, Value tensors (permuted)
            'Q2': (self.M, self.H, self.D),
            'K2': (self.M, self.H, self.D),
            'V2': (self.M, self.H, self.D),

            # Output tensors
            'O': (self.H, self.M, self.D),
            'O1': (self.M, self.H, self.D),
            'O2': (self.M, self.N),

            # Attention computation tensors
            'C': (self.H, self.M, self.M),
            'C_exp': (self.H, self.M, self.M),
            'C_div': (self.H, self.M, self.M),
            'C_sum': (self.H, self.M)
        }

    def get_backward_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get tensor shapes for backward pass (includes forward + gradients).

        Returns:
            Dictionary mapping tensor names to their shapes (tuples of ints)
        """
        shapes = self.get_forward_tensor_shapes().copy()

        # Add gradient tensors
        backward_specific = {
            # Weight gradients
            'dWQ': (self.N, self.N),
            'dWK': (self.N, self.N),
            'dWV': (self.N, self.N),

            # Query, Key, Value gradients (3D)
            'dQ': (self.H, self.M, self.D),
            'dK': (self.H, self.M, self.D),
            'dV': (self.H, self.M, self.D),

            # Query, Key, Value gradients (2D)
            'dQ1': (self.M, self.N),
            'dK1': (self.M, self.N),
            'dV1': (self.M, self.N),

            # Cache tensors
            'K_cache': (self.H, self.P + self.M, self.D),
            'V_cache': (self.H, self.P + self.M, self.D),

            # Output gradients
            'dO': (self.H, self.M, self.D),
            'dO_tmp': (self.H, self.M, self.D),
            'dO2': (self.M, self.N),

            # Attention gradients
            'dC': (self.H, self.M, self.M),
            'dC_exp': (self.H, self.M, self.M),
            'dC_sum': (self.H, self.M),

            # Perturbation tensors
            'noise': (self.H, self.M, self.P + self.M),
            'C_perturb': (self.H, self.M, self.P + self.M),
            'C_exp_perturb': (self.H, self.M, self.P + self.M),
            'C_sum_perturb': (self.H, self.M, self.P + self.M),
            'C_div_perturb': (self.H, self.M, self.P + self.M),
            'C_out': (self.H, self.P + self.M),
            'C_out1': (self.H, self.P + self.M),
            'C_out2': (self.H, self.P + self.M),

            # Normalized tensors
            'Q_norm': (self.H, self.M, self.D),
            'K_norm': (self.H, self.M, self.D)
        }

        shapes.update(backward_specific)
        return shapes

    @staticmethod
    def get_forward_shape_dict() -> Dict[str, Tuple[str, ...]]:
        """Get symbolic shape dictionary for forward pass.

        Returns:
            Dictionary mapping tensor names to their symbolic shapes (tuples of strings)
        """
        return {
            # Input and normalized tensors
            'X': ('M', 'N'),
            'X2': ('M', ),
            'X_norm': ('M', 'N'),

            # Weight matrices
            'WQ': ('N', 'N'),
            'WK': ('N', 'N'),
            'WV': ('N', 'N'),

            # Query, Key, Value tensors (3D)
            'Q': ('H', 'M', 'D'),
            'K': ('H', 'M', 'D'),
            'V': ('H', 'M', 'D'),

            # Query, Key, Value tensors (2D)
            'Q1': ('M', 'N'),
            'K1': ('M', 'N'),
            'V1': ('M', 'N'),

            # Query, Key, Value tensors (permuted)
            'Q2': ('M', 'H', 'D'),
            'K2': ('M', 'H', 'D'),
            'V2': ('M', 'H', 'D'),

            # Output tensors
            'O': ('H', 'M', 'D'),
            'O1': ('M', 'H', 'D'),
            'O2': ('M', 'N'),

            # Attention computation tensors
            'C': ('H', 'M', 'M'),
            'C_exp': ('H', 'M', 'M'),
            'C_div': ('H', 'M', 'M'),
            'C_sum': ('H', 'M')
        }

    @staticmethod
    def get_backward_shape_dict() -> Dict[str, Tuple[str, ...]]:
        """Get symbolic shape dictionary for backward pass.

        Returns:
            Dictionary mapping tensor names to their symbolic shapes (tuples of strings)
        """
        shapes = TensorShapeBuilder.get_forward_shape_dict().copy()

        # Add gradient tensors
        backward_specific = {
            # Weight gradients
            'dWQ': ('N', 'N'),
            'dWK': ('N', 'N'),
            'dWV': ('N', 'N'),

            # Query, Key, Value gradients (3D)
            'dQ': ('H', 'M', 'D'),
            'dK': ('H', 'M', 'D'),
            'dV': ('H', 'M', 'D'),

            # Query, Key, Value gradients (2D)
            'dQ1': ('M', 'N'),
            'dK1': ('M', 'N'),
            'dV1': ('M', 'N'),

            # Cache tensors
            'K_cache': ('H', 'P+M', 'D'),
            'V_cache': ('H', 'P+M', 'D'),

            # Output gradients
            'dO': ('H', 'M', 'D'),
            'dO_tmp': ('H', 'M', 'D'),
            'dO2': ('M', 'N'),

            # Attention gradients
            'dC': ('H', 'M', 'M'),
            'dC_exp': ('H', 'M', 'M'),
            'dC_sum': ('H', 'M'),

            # Perturbation tensors
            'noise': ('H', 'M', 'P+M'),
            'C_perturb': ('H', 'M', 'P+M'),
            'C_exp_perturb': ('H', 'M', 'P+M'),
            'C_sum_perturb': ('H', 'M', 'P+M'),
            'C_div_perturb': ('H', 'M', 'P+M'),
            'C_out': ('H', 'P+M'),
            'C_out1': ('H', 'P+M'),
            'C_out2': ('H', 'P+M'),

            # Normalized tensors
            'Q_norm': ('H', 'M', 'D'),
            'K_norm': ('H', 'M', 'D')
        }

        shapes.update(backward_specific)
        return shapes


# Convenience functions for backward compatibility
def get_forward_shapes(M: int, N: int, D: int, H: int, P: int) -> Dict[str, Tuple[int, ...]]:
    """Get forward pass tensor shapes.

    Args:
        M: Batch size
        N: Feature dimension
        D: Head dimension
        H: Number of heads
        P: Cache length

    Returns:
        Dictionary of tensor shapes
    """
    builder = TensorShapeBuilder(M, N, D, H, P)
    return builder.get_forward_tensor_shapes()


def get_backward_shapes(M: int, N: int, D: int, H: int, P: int) -> Dict[str, Tuple[int, ...]]:
    """Get backward pass tensor shapes.

    Args:
        M: Batch size
        N: Feature dimension
        D: Head dimension
        H: Number of heads
        P: Cache length

    Returns:
        Dictionary of tensor shapes
    """
    builder = TensorShapeBuilder(M, N, D, H, P)
    return builder.get_backward_tensor_shapes()


def get_forward_shape_dict() -> Dict[str, Tuple[str, ...]]:
    """Get forward pass symbolic shape dictionary.

    Returns:
        Dictionary of symbolic shapes
    """
    return TensorShapeBuilder.get_forward_shape_dict()


def get_backward_shape_dict() -> Dict[str, Tuple[str, ...]]:
    """Get backward pass symbolic shape dictionary.

    Returns:
        Dictionary of symbolic shapes
    """
    return TensorShapeBuilder.get_backward_shape_dict()
