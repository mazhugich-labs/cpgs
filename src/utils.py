"""Oscillator Coupling Matrix Utilities

This module provides utility functions to construct **coupling bias** and **coupling weight matrices**
for oscillator-based systems such as Central Pattern Generators (CPGs) used in rhythmic control
of legged robots.

Functions
---------
- make_coupling_bias_matrix(alpha)
- make_coupling_weight_matrix(coupling_bias, ...)
"""

import torch


@torch.jit.script
def make_coupling_bias_matrix(alpha: torch.Tensor) -> torch.Tensor:
    r"""Compute the **inter-oscillator coupling bias matrix**.

    Each entry :math:`B_{ij}` in the resulting matrix represents the signed
    phase difference between oscillator *i* and oscillator *j*:

    .. math::
        {B}_{ij} = \alpha_i - \alpha_j

    Args:
        alpha (torch.Tensor):
            1D tensor of shape ``(N,)`` containing the per-oscillator phase values (in radians).

    Returns:
        torch.Tensor:
            Tensor of shape ``(N, N)`` representing pairwise phase differences between oscillators.

    Example:
        >>> alpha = torch.tensor([0.0, 1.0, 2.0])
        >>> make_coupling_bias_matrix(alpha)
        tensor([[ 0., -1., -2.],
                [ 1.,  0., -1.],
                [ 2.,  1.,  0.]])

    Notes:
        - The resulting matrix is **antisymmetric**: :math:`B_{ij} = -B_{ji}`.
        - The diagonal entries are always zero.
    """
    return alpha.unsqueeze(0) - alpha.unsqueeze(1)


@torch.jit.script
def make_coupling_weight_matrix(
    coupling_bias: torch.Tensor,
    self_coupling_weight: float = 0.0,
    in_group_coupling_weight: float = 1.0,
    of_group_coupling_weight: float = 0.0,
    threshold: float = 1e-3,
) -> torch.Tensor:
    r"""Compute the **inter-oscillator coupling weight matrix** from a given coupling bias matrix.

    Each element :math:`W_{ij}` of the output matrix is determined by comparing the
    phase bias :math:`|B_{ij}|` against a given threshold:

    .. math::
        W_{ij} =
        \begin{cases}
        w_\text{self} & \text{if } i = j \\
        w_\text{in}   & \text{if } |B_{ij}| \leq \text{threshold} \\
        w_\text{off}  & \text{otherwise}
        \end{cases}

    Args:
        coupling_bias (torch.Tensor):
            Tensor of shape ``(N, N)`` representing pairwise phase differences between oscillators.

        self_coupling_weight (float, optional):
            Weight assigned to self-coupling terms (diagonal elements).
            Defaults to ``0.0``.

        in_group_coupling_weight (float, optional):
            Weight applied between oscillators whose phase difference is within the given threshold.
            Defaults to ``1.0``.

        of_group_coupling_weight (float, optional):
            Weight applied between oscillators considered out-of-group (different phase groups).
            Defaults to ``0.0``.

        threshold (float, optional):
            Maximum absolute phase difference (in radians) for two oscillators to be considered in the same group.
            Defaults to ``1e-3``.

    Returns:
        torch.Tensor:
            Coupling weight matrix of shape ``(N, N)``.

    Example:
        >>> alpha = torch.tensor([0.0, 0.0, 1.0])
        >>> bias = make_coupling_bias_matrix(alpha)
        >>> make_coupling_weight_matrix(
        ...     bias,
        ...     self_coupling_weight=0.0,
        ...     in_group_coupling_weight=1.0,
        ...     of_group_coupling_weight=0.2,
        ...     threshold=1e-2,
        ... )
        tensor([[0.0000, 1.0000, 0.2000],
                [1.0000, 0.0000, 0.2000],
                [0.2000, 0.2000, 0.0000]])

    Notes:
        - The output matrix is typically **symmetric**, since phase differences are symmetric.
    """
    coupling_weight = torch.full_like(coupling_bias, of_group_coupling_weight)

    coupling_weight[(coupling_bias >= -threshold) & (coupling_bias <= threshold)] = (
        in_group_coupling_weight
    )

    return coupling_weight.fill_diagonal_(self_coupling_weight)
