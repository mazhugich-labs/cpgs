from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


@dataclass
class BaseOscillatorCfg(ABC):
    @dataclass
    class InitialStateCfg(ABC):
        pass

    init_state: InitialStateCfg

    dt: float

    def __post__init__(self):
        assert self.dt > 0, ""


@dataclass
class BaseOscillatorData(ABC):
    @dataclass
    class DefaultStateCfg(ABC):
        pass

    default_state: DefaultStateCfg


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
