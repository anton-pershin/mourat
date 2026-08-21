from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mourat.monitoring import MonitoringHandler

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Function(Generic[InputT, OutputT], ABC):
    def __init__(self, monitoring_handler: MonitoringHandler) -> None:
        self.monitoring_handler = monitoring_handler

    def __call__(self, data: InputT, step_id: str) -> OutputT:
        output, text_for_monitoring = self._run(data)
        self.monitoring_handler(step=step_id, text_for_monitoring=text_for_monitoring)
        return output

    @abstractmethod
    def _run(self, data: InputT) -> tuple[OutputT, str]:
        raise NotImplementedError()
