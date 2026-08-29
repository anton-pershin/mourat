import logging
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mourat.monitoring import MonitoringHandler

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Function(Generic[InputT, OutputT], ABC):
    def __init__(self, monitoring_handler: MonitoringHandler) -> None:
        self.monitoring_handler = monitoring_handler

    def __call__(self, data: InputT, step_id: str) -> OutputT:
        name = type(self).__name__
        logger.info("step %s | %s | start", step_id, name)

        t0 = time.monotonic()
        try:
            output, text_for_monitoring = self._run(data)
        except Exception:
            logger.exception(
                "step %s | %s | FAILED after %.2fs",
                step_id,
                name,
                time.monotonic() - t0,
            )
            raise
        run_s = time.monotonic() - t0

        t1 = time.monotonic()
        self.monitoring_handler(step=step_id, text_for_monitoring=text_for_monitoring)
        monitoring_s = time.monotonic() - t1

        logger.info(
            "step %s | %s | done | run=%.2fs monitoring=%.2fs total=%.2fs",
            step_id,
            name,
            run_s,
            monitoring_s,
            run_s + monitoring_s,
        )
        return output

    @abstractmethod
    def _run(self, data: InputT) -> tuple[OutputT, str]:
        raise NotImplementedError()
