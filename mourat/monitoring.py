import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@abstractmethod
class MonitoringHandler(ABC):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        step: str,
        text_for_monitoring: str,
        data: Any | None = None,
    ) -> None:
        raise NotImplementedError

    def start_step(self, step: str) -> None:
        pass

    def append_data(self, step: str, data: Any) -> None:
        pass


class MonitoringViaMarkdownFiles(MonitoringHandler):
    def __init__(self, filename_template: str) -> None:
        # Should contain exactly one key: step
        self.filename_template = filename_template

    def __call__(
        self,
        step: str,
        text_for_monitoring: str,
        data: Any | None = None,
    ) -> None:
        filename = self.filename_template.format(step=step)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text_for_monitoring)


class MonitoringViaJsonlFiles(MonitoringHandler):
    def __init__(self, filename_template: str) -> None:
        # Should contain exactly one key: step
        self.filename_template = filename_template

    def __call__(
        self,
        step: str,
        text_for_monitoring: str,
        data: Any | None = None,
    ) -> None:
        filename = Path(self.filename_template.format(step=step))
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            for record in self._to_records(data):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start_step(self, step: str) -> None:
        filename = Path(self.filename_template.format(step=step))
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("", encoding="utf-8")

    def append_data(self, step: str, data: Any) -> None:
        filename = Path(self.filename_template.format(step=step))
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "a", encoding="utf-8") as f:
            for record in self._to_records(data):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def _to_records(self, data: Any | None) -> list[dict[str, Any]]:
        if data is None:
            return []

        if isinstance(data, BaseModel):
            collection_records = self._collection_model_to_records(data)
            if collection_records is not None:
                return collection_records

            return [data.model_dump(mode="json", exclude_none=True)]

        if isinstance(data, list):
            return [self._to_record(item) for item in data]

        return [self._to_record(data)]

    def _collection_model_to_records(
        self,
        data: BaseModel,
    ) -> list[dict[str, Any]] | None:
        dumped = data.model_dump(mode="json", exclude_none=True)
        list_fields = [
            field_value
            for field_value in dumped.values()
            if isinstance(field_value, list)
        ]

        if len(dumped) == 1 and len(list_fields) == 1:
            return [self._to_record(item) for item in list_fields[0]]

        return None

    @staticmethod
    def _to_record(data: Any) -> dict[str, Any]:
        if isinstance(data, BaseModel):
            return data.model_dump(mode="json", exclude_none=True)

        if isinstance(data, dict):
            return data

        return {"value": data}
