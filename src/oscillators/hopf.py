r"""Hopf Oscillator Implementation

This module implements the **Hopf Oscillator**, a nonlinear dynamical system commonly used
as the foundational element in **Central Pattern Generators (CPGs)** for rhythmic control
of legged robots.

It provides:
    - Structured configuration and data classes via dataclasses
    - A `HopfOscillator` class that handles state integration and coupling
    - An optional `HopfAdapter` class that maps control inputs to oscillator parameters

The implementation follows the standard Hopf oscillator equations with coupling dynamics:

.. math::
    \dot{r}_i = v_i \\
    \dot{v}_i = \frac{\mu^2}{4}(r_i^{cmd} - r_i) - \mu v_i \\
    \dot{\alpha}_i = \Delta \theta_i \\
    \dot{\beta}_i = \frac{\Delta \theta_{max}}{2} + \sum_j w_{ij} \sin(\beta_j - \beta_i - \phi_{ij})

where:
    - :math:`r_i` is the oscillator amplitude
    - :math:`\alpha_i` and :math:`\beta_i` are modulated and gait-specific phase components
    - :math:`\phi_{ij}` is the phase coupling bias between oscillators *i* and *j*
    - :math:`w_{ij}` is the coupling weight matrix
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .abc import (
    BaseOscillatorCfg,
    BaseOscillatorData,
    BaseOscillator,
    BaseAdapterCfg,
    BaseAdapter,
)


@dataclass
class HopfOscillatorCfg(BaseOscillatorCfg):
    """Configuration class for the Hopf Oscillator

    Attributes:
        init_state (HopfOscillatorCfg.InitialStateCfg):
            Nested configuration specifying the initial per-oscillator states.
        mu (float):
            Convergence factor controlling the rate of amplitude convergence.
    """

    @dataclass
    class InitialStateCfg(BaseOscillatorCfg.InitialStateCfg):
        """Initial state configuration for the Hopf oscillator.

        Attributes:
            beta (Sequence[float]):
                Initial per-oscillator phase values in radians.

        Raises:
            AssertionError: if fewer than 2 oscillators are defined.
        """

        beta: Sequence[float]

        def __post_init__(self):
            assert len(self.beta) > 1, "There must be at least 2 oscillators"

    init_state: InitialStateCfg
    """Initial state configuration"""

    mu: float
    """Convergence factor"""


@dataclass
class HopfOscillatorData(BaseOscillatorData):
    """Data container for Hopf oscillator runtime state variables"""

    @dataclass
    class DefaultStateCfg(BaseOscillatorData.DefaultStateCfg):
        """Default (reset) state configuration"""

        beta: torch.Tensor

    default_state: DefaultStateCfg
    """Default state configuration"""

    r: torch.Tensor
    delta_r: torch.Tensor
    """Position and linear velocity"""

    v: torch.Tensor
    delta_v: torch.Tensor
    """Linear velocity and acceleration"""

    alpha: torch.Tensor
    delta_alpha: torch.Tensor
    """Modulated phase and its rate"""

    beta: torch.Tensor
    delta_beta: torch.Tensor
    """Gait-specific phase and its rate"""

    @property
    def theta(self) -> torch.Tensor:
        """Oscillator mixed phase"""
        return torch.remainder(self.alpha + self.beta, 2 * math.pi)

    @property
    def delta_theta(self) -> torch.Tensor:
        """Oscillator mixed frequency"""
        return self.delta_alpha + self.delta_beta


class HopfOscillator(BaseOscillator):
    """Implementation of a vectorized Hopf oscillator network with inter-oscillator coupling"""

    _cfg: HopfOscillatorCfg
    _data: HopfOscillatorData

    def __init__(self, cfg: HopfOscillatorCfg, device: str = "cpu"):
        """Initialize Hopf oscillator with given configuration"""
        self._cfg = cfg

        default_beta = torch.tensor(self._cfg.init_state.beta, device=device)
        self._data = HopfOscillatorData(
            default_state=HopfOscillatorData.DefaultStateCfg(
                beta=default_beta,
            ),
            r=torch.zeros_like(default_beta),
            delta_r=torch.zeros_like(default_beta),
            v=torch.zeros_like(default_beta),
            delta_v=torch.zeros_like(default_beta),
            alpha=torch.zeros_like(default_beta),
            delta_alpha=torch.zeros_like(default_beta),
            beta=default_beta.clone(),
            delta_beta=torch.zeros_like(default_beta),
        )

    def _compute_coupling_term(
        self, coupling_bias: torch.Tensor, coupling_weight: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute inter-oscillator coupling term

        Args:
            coupling_bias (torch.Tensor): Phase coupling bias matrix φ_ij.
            coupling_weight (torch.Tensor): Coupling weight matrix w_ij.

        Returns:
            torch.Tensor: Coupling term per oscillator.
        """
        return torch.sum(
            self._data.r.unsqueeze(1)
            * coupling_weight
            * torch.sin(
                self._data.beta.unsqueeze(-2)
                - self._data.beta.unsqueeze(-1)
                - coupling_bias
            ),
            dim=1,
        )

    def _compute_delta_state(
        self,
        r: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute desired delta state using Hopf oscillator equations

        Args:
            r (torch.Tensor): Amplitude modulation command.
            delta_theta (torch.Tensor): Frequency modulation command.
            delta_theta_max (torch.Tensor): Frequency upper bound.
            coupling_bias (torch.Tensor): Coupling bias matrix.
            coupling_weight (torch.Tensor): Coupling weight matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Desired (Δr, Δv, Δα, Δβ) state.
        """
        return (
            self._data.v,
            self._cfg.mu**2 / 4 * (r - self._data.r) - self._cfg.mu * self._data.v,
            delta_theta,
            delta_theta_max / 2
            + self._compute_coupling_term(coupling_weight, coupling_bias),
        )

    def _compute_delta_state_heun(
        self,
        delta_r_desired: torch.Tensor,
        delta_v_desired: torch.Tensor,
        delta_alpha_desired: torch.Tensor,
        delta_beta_desired: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute target delta state using Heun’s integration method (improved Euler)

        This provides better numerical stability than explicit Euler integration.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Averaged (Δr, Δv, Δα, Δβ) targets.
        """
        return (
            (self._data.delta_r + delta_r_desired) / 2,
            (self._data.delta_v + delta_v_desired) / 2,
            (self._data.delta_alpha + delta_alpha_desired) / 2,
            (self._data.delta_beta + delta_beta_desired) / 2,
        )

    def _integrate(
        self,
        delta_r_target: torch.Tensor,
        delta_v_target: torch.Tensor,
        delta_alpha_target: torch.Tensor,
        delta_beta_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate oscillator state forward in time using configured Δt

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Updated (r, v, α, β) state tensors.
        """
        return (
            self._data.r + delta_r_target * self._cfg.dt,
            self._data.v + delta_v_target * self._cfg.dt,
            torch.remainder(
                self._data.alpha + delta_alpha_target * self._cfg.dt, 2 * math.pi
            ),
            torch.remainder(
                self._data.beta + delta_beta_target * self._cfg.dt, 2 * math.pi
            ),
        )

    def reset(self) -> None:
        """Reset oscillator state to its default configuration"""
        self._data.r[:] = 0.0
        self._data.delta_r[:] = 0.0
        self._data.v[:] = 0.0
        self._data.delta_v[:] = 0.0
        self._data.alpha[:] = 0.0
        self._data.delta_alpha[:] = 0.0
        self._data.beta[:] = self._data.default_state.beta[:]
        self._data.delta_beta[:] = 0.0

    def step(
        self,
        r: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> HopfOscillatorData:
        """Advance oscillator dynamics by one time step

        Args:
            r (torch.Tensor): Amplitude modulation input.
            delta_theta (torch.Tensor): Frequency modulation input.
            delta_theta_max (torch.Tensor): Maximum frequency output.
            coupling_bias (torch.Tensor): Coupling bias matrix.
            coupling_weight (torch.Tensor): Coupling weight matrix.

        Returns:
            HopfOscillatorData: Updated oscillator state data.
        """
        delta_state_desired = self._compute_delta_state(
            r,
            delta_theta,
            delta_theta_max,
            coupling_bias,
            coupling_weight,
        )

        delta_state_target = self._compute_delta_state_heun(*delta_state_desired)

        self._data.r, self._data.v, self._data.alpha, self._data.beta = self._integrate(
            *delta_state_target
        )

        (
            self._data.delta_r,
            self._data.delta_v,
            self._data.delta_alpha,
            self._data.delta_beta,
        ) = delta_state_target

        return self._data

    @property
    def cfg(self) -> HopfOscillatorCfg:
        """Return the Hopf oscillator configuration"""
        return self._cfg

    @property
    def data(self) -> HopfOscillatorData:
        """Return the current oscillator data"""
        return self._data


@dataclass
class HopfAdapterCfg(BaseAdapterCfg):
    """Adapter configuration for mapping control inputs to oscillator commands.

    Attributes:
        r_range (tuple[float, float]): Range of amplitude modulation values.
        delta_theta_min (float): Minimum allowed oscillator frequency.
    """

    r_range: list[float, float] | tuple[float, float]

    delta_theta_min: float


class HopfAdapter(BaseAdapter):
    """Adapter that decodes normalized control inputs into oscillator commands"""

    _cfg: HopfAdapterCfg
    _osc: HopfOscillator

    def _decode(
        self,
        r: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_theta_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode normalized input into physical oscillator parameters"""
        return (
            self._cfg.r_range[0]
            + (r - self._cfg.action_range[0])
            / (self._cfg.action_range[1] - self._cfg.action_range[0])
            * (self._cfg.r_range[1] - self._cfg.r_range[0]),
            self._cfg.delta_theta_min
            + (delta_theta - self._cfg.action_range[0])
            / (self._cfg.action_range[1] - self._cfg.action_range[0])
            * (delta_theta_max - self._cfg.delta_theta_min),
            delta_theta_max,
        )

    def __call__(
        self,
        r: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> HopfOscillatorData:
        """Compute one oscillator update step from decoded control inputs"""
        return self._osc.step(
            *self._decode(r, delta_theta, delta_theta_max),
            coupling_bias,
            coupling_weight
        )
