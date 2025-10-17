import torch


@torch.jit.script
def make_coupling_bias_matrix(alpha: torch.Tensor) -> torch.Tensor:
    """Compute inter-oscillator coupling bias matrix

    Args:
        alpha (torch.Tensor): initial per-oscillator phase

    Returns:
        torch.Tensor: coupling bias matrix
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
    """Compute inter-oscillator coupling weight matrix given coupling bias matrix

    Args:
        coupling_bias (torch.Tensor): coupling bias matrix
        self_coupling_weight (float, optional): selg-coupling weight. Defaults to 0.0.
        in_group_coupling_weight (float, optional): in-group coupling weight. Defaults to 1.0.
        of_group_coupling_weight (float, optional): off-group coupling weight. Defaults to 0.0.
        threshold (float, optional): difference threshold at which to count oscillators as a group. Defaults to 1e-3.

    Returns:
        torch.Tensor: coupling weight matrix
    """
    coupling_weight = torch.full_like(coupling_bias, of_group_coupling_weight)

    coupling_weight[(coupling_bias >= -threshold) & (coupling_bias <= threshold)] = (
        in_group_coupling_weight
    )

    return coupling_weight.fill_diagonal_(self_coupling_weight)
