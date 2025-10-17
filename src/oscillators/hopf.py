import math
from dataclasses import dataclass

import torch

from .abc import BaseOscillatorCfg, BaseOscillatorData, BaseOscillator


@dataclass
class HopfOscillatorCfg(BaseOscillatorCfg):
    @dataclass
    class InitialStateCfg(BaseOscillatorCfg.InitialStateCfg):
        beta: tuple[float, ...]
        """Initial per-oscillator phase"""

        def __post_init__(self):
            assert len(self.beta) > 1, ""

    init_state: InitialStateCfg
    """Initial state configuration"""

    mu: float
    """Convergence factor"""


@dataclass
class HopfOscillatorData(BaseOscillatorData):
    @dataclass
    class DefaultStateCfg(BaseOscillatorData.DefaultStateCfg):
        beta: torch.Tensor
        """Default per-oscillator phase"""

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
    _cfg: HopfOscillatorCfg
    _data: HopfOscillatorData

    def __init__(self, cfg: HopfOscillatorCfg, device: str) -> None:
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
            beta=default_beta,
            delta_beta=torch.zeros_like(default_beta),
        )

    def _compute_coupling_term(
        self, coupling_bias: torch.Tensor, coupling_weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute inter-oscillator coupling term

        Args:
            coupling_bias (torch.Tensor): coupling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix

        Returns:
            torch.Tensor: inter-oscillator coupling term
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
        r_cmd: torch.Tensor,
        delta_theta_cmd: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute desired delta state using Hopf oscillator equations

        Args:
            r_cmd (torch.Tensor): magnitude modulation command
            delta_theta_cmd (torch.Tensor): frequency  modulation command
            delta_theta_max (torch.Tensor): frequency output upper bound
            coupling_bias (torch.Tensor): counpling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: desired delta state
        """
        return (
            self._data.v,
            self._cfg.mu**2 / 4 * (r_cmd - self._data.r) - self._cfg.mu * self._data.v,
            delta_theta_cmd,
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
        """Compute target delta state using Heun method

        Args:
            delta_r_desired (torch.Tensor): desired velocity
            delta_v_desired (torch.Tensor): desired acceleration
            delta_alpha_desired (torch.Tensor): desired modulated frequency
            delta_beta_desired (torch.Tensor): desired gait-specific frequency

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: target delta state
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
        """Compute new oscillator state

        Args:
            delta_r_target (torch.Tensor): target velocity
            delta_v_target (torch.Tensor): target accelleration
            delta_alpha_target (torch.Tensor): target modulated frequency
            delta_beta_target (torch.Tensor): target gait-specific frequency

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: new oscillator state
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
        """Reset oscillator state to defaults"""
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
        r_cmd: torch.Tensor,
        delta_theta_cmd: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> HopfOscillatorData:
        """Preprocess input and produce new oscillator state

        Args:
            r_cmd (torch.Tensor): oscillator magnitude modulation command
            delta_theta_cmd (torch.Tensor): oscillator frequency modulation command
            delta_theta_max (torch.Tensor): oscillator frequency upper bound
            coupling_bias (torch.Tensor): coupling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix

        Returns:
            tuple[torch.Tensor, torch.Tensor]: new oscillator magnitude and phase
        """
        delta_state_desired = self._compute_delta_state(
            r_cmd,
            delta_theta_cmd,
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
        """Hopf oscillator configuration

        Returns:
            HopfOscillatorCfg: configuration
        """
        return self._cfg

    @property
    def data(self) -> HopfOscillatorData:
        """Hopf oscillator data

        Returns:
            HopfOscillatorData: data
        """
        return self._data
