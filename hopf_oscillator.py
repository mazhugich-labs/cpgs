import math
from dataclasses import dataclass, MISSING

import torch


# ---------- Configuration ----------
@dataclass
class HopfOscillatorCfg:
    @dataclass
    class InitialStateCfg:
        beta: tuple[float] | list[float] = MISSING
        """Initial per-oscillator phase"""

    init_state: InitialStateCfg = MISSING
    """Initial state configuration"""

    mu: float = 32
    """Convergence factor"""


# ---------- Data ----------
@dataclass
class HopfOscillatorData:
    @dataclass
    class DefaultStateCfg:
        beta: torch.Tensor = MISSING
        """Default per-oscillator phase"""

    default_state: DefaultStateCfg = MISSING
    """Default state configuration"""

    r: torch.Tensor = None
    delta_r: torch.Tensor = None
    """Position and linear velocity"""

    v: torch.Tensor = None
    delta_v: torch.Tensor = None
    """Linear velocity and acceleration"""

    alpha: torch.Tensor = None
    delta_alpha: torch.Tensor = None
    """Modulated phase and its rate"""

    beta: torch.Tensor = None
    delta_beta: torch.Tensor = None
    """Gait-specific phase and its rate"""

    @property
    def theta(self) -> torch.Tensor:
        """Oscillator mixed phase"""
        return torch.remainder(self.alpha + self.beta, 2 * math.pi)

    @property
    def delta_theta(self) -> torch.Tensor:
        """Oscillator mixed frequency"""
        return self.delta_alpha + self.delta_beta


# ---------- Module ----------
class HopfOscillator:
    _cfg: HopfOscillatorCfg
    _data: HopfOscillatorData

    def __init__(self, cfg: HopfOscillatorCfg, num_envs: int, device: str) -> None:
        self._cfg = cfg

        self._data = HopfOscillatorData(
            default_state=HopfOscillatorData.DefaultStateCfg(
                beta=torch.tensor(self._cfg.init_state.beta, device=device)
            )
        )

        self._data.r = torch.zeros_like(self._data.default_state.beta).repeat(
            num_envs, 1
        )
        self._data.delta_r = torch.zeros_like(self._data.r)
        self._data.v = torch.zeros_like(self._data.r)
        self._data.delta_v = torch.zeros_like(self._data.r)
        self._data.alpha = torch.zeros_like(self._data.r)
        self._data.delta_alpha = torch.zeros_like(self._data.r)
        self._data.beta = self._data.default_state.beta.repeat(num_envs, 1)
        self._data.delta_beta = torch.zeros_like(self._data.r)

    # ---------- Intrinsic methods ----------
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
                self._data.beta.unsqueeze(1)
                - self._data.beta.unsqueeze(2)
                - coupling_bias
            ),
            dim=1,
        )

    def _compute_delta_state_desired(
        self,
        r_cmd: torch.Tensor,
        delta_theta_cmd: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute desired delta state using Hopf oscillator equations

        Args:
            r_cmd (torch.Tensor): magnitude modulation command
            delta_theta_cmd (torch.Tensor): frequency  modulation command
            delta_theta_max (torch.Tensor): frequency output upper bound
            coupling_bias (torch.Tensor): counpling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix

        Returns:
            tuple[torch.Tensor]: desired delta state
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
    ) -> tuple[torch.Tensor]:
        """Compute target delta state using Heun method

        Args:
            delta_r_desired (torch.Tensor): desired velocity
            delta_v_desired (torch.Tensor): desired acceleration
            delta_alpha_desired (torch.Tensor): desired modulated frequency
            delta_beta_desired (torch.Tensor): desired gait-specific frequency

        Returns:
            tuple[torch.Tensor]: target delta state
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
        dt: float,
    ) -> tuple[torch.Tensor]:
        """Compute new oscillator state

        Args:
            delta_r_target (torch.Tensor): target velocity
            delta_v_target (torch.Tensor): target accelleration
            delta_alpha_target (torch.Tensor): target modulated frequency
            delta_beta_target (torch.Tensor): target gait-specific frequency
            dt (float): time step

        Returns:
            tuple[torch.Tensor]: new oscillator state
        """
        return (
            self._data.r + delta_r_target * dt,
            self._data.v + delta_v_target * dt,
            torch.remainder(self._data.alpha + delta_alpha_target * dt, 2 * math.pi),
            torch.remainder(self._data.beta + delta_beta_target * dt, 2 * math.pi),
        )

    # ---------- Public API ----------
    def reset(self, env_ids: tuple[int] | list[int]) -> None:
        """Reset oscillator state to defaults

        Args:
            env_ids (tuple[int] | list[int]): environment indicies
        """
        self._data.r[env_ids] = 0.0
        self._data.delta_r[env_ids] = 0.0
        self._data.v[env_ids] = 0.0
        self._data.delta_v[env_ids] = 0.0
        self._data.alpha[env_ids] = 0.0
        self._data.delta_alpha[env_ids] = 0.0
        self._data.beta[env_ids] = self._data.default_state.beta
        self._data.delta_beta[env_ids] = 0.0

    def step(
        self,
        r_cmd: torch.Tensor,
        delta_theta_cmd: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
        dt: float,
    ) -> None:
        """Preprocess input and produce new oscillator state

        Args:
            r_cmd (torch.Tensor): oscillator magnitude modulation command
            delta_theta_cmd (torch.Tensor): oscillator frequency modulation command
            delta_theta_max (torch.Tensor): oscillator frequency upper bound
            coupling_bias (torch.Tensor): coupling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix
            dt (float): time step

        Returns:
            tuple[torch.Tensor, torch.Tensor]: new oscillator magnitude and phase
        """
        delta_state_desired = self._compute_delta_state_desired(
            r_cmd,
            delta_theta_cmd,
            delta_theta_max,
            coupling_bias,
            coupling_weight,
        )

        delta_state_target = self._compute_delta_state_heun(*delta_state_desired)

        self._data.r, self._data.v, self._data.alpha, self._data.beta = self._integrate(
            *delta_state_target, dt
        )

        (
            self._data.delta_r,
            self._data.delta_v,
            self._data.delta_alpha,
            self._data.delta_beta,
        ) = delta_state_target

    # ---------- Dunder methods ----------
    def __call__(
        self,
        r_cmd: torch.Tensor,
        delta_theta_cmd: torch.Tensor,
        delta_theta_max: torch.Tensor,
        coupling_bias: torch.Tensor,
        coupling_weight: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess input and produce new oscillator state

        Args:
            r_cmd (torch.Tensor): oscillator magnitude modulation command
            delta_theta_cmd (torch.Tensor): oscillator frequency modulation command
            delta_theta_max (torch.Tensor): oscillator frequency upper bound
            coupling_bias (torch.Tensor): coupling bias matrix
            coupling_weight (torch.Tensor): coupling weight matrix
            dt (float): time step

        Returns:
            tuple[torch.Tensor, torch.Tensor]: new oscillator magnitude and phase
        """
        return self.step(
            r_cmd,
            delta_theta_cmd,
            delta_theta_max,
            coupling_bias,
            coupling_weight,
            dt,
        )

    # ---------- Properties ----------
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
