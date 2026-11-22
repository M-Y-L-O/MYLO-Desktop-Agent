from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ResearchRunStore:
    SCHEMA_VERSION = "mylo.research-run.v1"

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)

    def save(
        self,
        *,
        run_type: str,
        project_id: str,
        request: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        os.makedirs(self.directory, exist_ok=True)
        run_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": run_id,
            "run_type": str(run_type),
            "project_id": str(project_id or ""),
            "created_at": created_at,
            "request": request,
            "result": result,
        }
        target = self._path(run_id)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{run_id}.", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, default=_json_default)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
        return {
            "run_id": run_id,
            "run_type": str(run_type),
            "created_at": created_at,
            "path": target,
        }

    def load(self, run_id: str) -> Dict[str, Any]:
        path = self._path(run_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Research run '{run_id}' does not exist.")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list(self, *, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not os.path.isdir(self.directory):
            return []
        records = []
        for filename in os.listdir(self.directory):
            if not filename.endswith(".json"):
                continue
            run_id = filename[:-5]
            if not _RUN_ID_PATTERN.fullmatch(run_id):
                continue
            try:
                record = self.load(run_id)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if project_id is not None and record.get("project_id") != project_id:
                continue
            records.append(
                {
                    "run_id": record.get("run_id"),
                    "run_type": record.get("run_type"),
                    "project_id": record.get("project_id"),
                    "created_at": record.get("created_at"),
                }
            )
        return sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)

    def _path(self, run_id: str) -> str:
        normalized = str(run_id).strip().lower()
        if not _RUN_ID_PATTERN.fullmatch(normalized):
            raise ValueError("run_id must be a 32-character lowercase hexadecimal ID.")
        return os.path.join(self.directory, f"{normalized}.json")


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)
