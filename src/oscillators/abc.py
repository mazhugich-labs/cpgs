from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


@dataclass
class BaseOscillatorCfg(ABC):
    @dataclass
    class InitialStateCfg(ABC):
        pass

    init_state: InitialStateCfg
    """Initial state configuration"""

    dt: float
    """Update rate"""

    def __post__init__(self):
        assert self.dt > 0, "Update rate must be greater than 0"


@dataclass
class BaseOscillatorData(ABC):
    @dataclass
    class DefaultStateCfg(ABC):
        pass

    default_state: DefaultStateCfg
    """Default state configuration"""


class BaseOscillator(ABC):
    _cfg: BaseOscillatorCfg
    _data: BaseOscillatorData

    def __init__(self, cfg: BaseOscillatorCfg):
        self._cfg = cfg

    @abstractmethod
    def _compute_delta_state(self, *args, **kwargs) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def _integrate(self, *args, **kwargs) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, *args, **kwargs) -> BaseOscillatorData: ...

    @property
    def cfg(self) -> BaseOscillatorCfg:
        return self._cfg

    @property
    def data(self) -> BaseOscillatorData:
        return self._data


@dataclass
class BaseAdapterCfg(ABC):
    action_range: list[float, float] | tuple[float, float]
    """Action/input range"""


class BaseAdapter(ABC):
    _cfg: BaseAdapterCfg
    _osc: BaseOscillator

    def __init__(self, cfg: BaseAdapterCfg, osc: BaseOscillator):
        self._osc = osc
        self._cfg = cfg

    @abstractmethod
    def _decode(self, *args, **kwargs) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> BaseOscillatorData: ...

    @property
    def cfg(self) -> BaseAdapterCfg:
        return self._cfg

    @property
    def osc(self) -> BaseOscillator:
        return self._osc
