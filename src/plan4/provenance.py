from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import load_dataset_manifest, load_literature_registry
from .models import RunPaths


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def plan_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "Plan 4":
            return parent
    raise RuntimeError("Unable to locate Plan 4 root from package path.")


def ensure_run_paths(run_id: str, output_root: str | Path | None = None) -> RunPaths:
    root = Path(output_root) if output_root is not None else plan_root()
    raw_results_dir = root / "results" / "raw"
    processed_results_dir = root / "results" / "processed"
    figures_dir = root / "results" / "figures"
    logs_dir = root / "logs" / "runs"

    for directory in [raw_results_dir, processed_results_dir, figures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        output_root=root,
        raw_results_dir=raw_results_dir,
        processed_results_dir=processed_results_dir,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
        summary_path=raw_results_dir / f"{run_id}_summary.json",
        trace_path=logs_dir / f"{run_id}_trace.jsonl",
        manifest_path=logs_dir / f"{run_id}_manifest.json",
    )


def json_dump(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def verify_dataset_manifest(manifest_path: str | Path) -> dict[str, Any]:
    entries = load_dataset_manifest(manifest_path)
    results = []
    valid = True
    for entry in entries:
        file_path = Path(entry.source_path)
        exists = file_path.exists()
        actual_hash = sha256_file(file_path) if exists else None
        hash_matches = actual_hash == entry.sha256 if exists else False
        valid = valid and exists and hash_matches
        results.append(
            {
                "data_id": entry.data_id,
                "exists": exists,
                "hash_matches": hash_matches,
                "expected_sha256": entry.sha256,
                "actual_sha256": actual_hash,
                "source_path": entry.source_path,
            }
        )
    return {"valid": valid, "items": results}


def verify_literature_registry(registry_path: str | Path) -> dict[str, Any]:
    entries = load_literature_registry(registry_path)
    results = []
    valid = True
    for entry in entries:
        if entry.local_pdf:
            file_path = Path(registry_path).parent / entry.local_pdf if not Path(entry.local_pdf).is_absolute() else Path(entry.local_pdf)
            exists = file_path.exists()
            actual_hash = sha256_file(file_path) if exists else None
            hash_matches = actual_hash == entry.sha256 if exists and entry.sha256 else None
        else:
            exists = False
            actual_hash = None
            hash_matches = None
        if entry.local_pdf and (not exists or (entry.sha256 and actual_hash != entry.sha256)):
            valid = False
        results.append(
            {
                "lit_id": entry.lit_id,
                "has_local_pdf": bool(entry.local_pdf),
                "exists": exists,
                "hash_matches": hash_matches,
                "verification_level": entry.verification_level,
                "doi": entry.doi,
                "verify_url": entry.verify_url,
            }
        )
    return {"valid": valid, "items": results}
