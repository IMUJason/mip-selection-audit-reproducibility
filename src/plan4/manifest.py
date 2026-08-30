from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetEntry:
    data_id: str
    group: str
    instance_name: str
    filename: str
    source_path: str
    source_collection: str
    sha256: str
    notes: str


@dataclass
class LiteratureEntry:
    lit_id: str
    authors_year: str
    title: str
    local_pdf: str | None
    sha256: str | None
    doi: str | None
    verify_url: str | None
    verification_level: str


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_manifest(path: str | Path) -> list[DatasetEntry]:
    payload = load_json(path)
    return [DatasetEntry(**item) for item in payload["items"]]


def load_literature_registry(path: str | Path) -> list[LiteratureEntry]:
    payload = load_json(path)
    return [LiteratureEntry(**item) for item in payload["items"]]


def find_dataset_entry(entries: list[DatasetEntry], data_id: str) -> DatasetEntry:
    for entry in entries:
        if entry.data_id == data_id:
            return entry
    raise KeyError(f"Dataset id not found: {data_id}")
