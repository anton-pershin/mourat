from abc import ABC, abstractmethod


@abstractmethod
class MonitoringHandler(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        raise NotImplementedError


class MonitoringViaMarkdownFiles(MonitoringHandler):
    def __init__(self, filename_template: str) -> None:
        # Should contain exactly one key: step
        self.filename_template = filename_template

    def __call__(self, step: str, text_for_monitoring: str) -> None:
        filename = self.filename_template.format(step=step)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text_for_monitoring)
