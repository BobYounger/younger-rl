from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


class MetricLogger:
    """Persist scalar training metrics to JSONL and CSV."""

    def __init__(
        self,
        log_dir: str | Path,
        run_name: Optional[str] = None,
        csv_name: str = "metrics.csv",
        jsonl_name: str = "metrics.jsonl",
    ) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or timestamp
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / csv_name
        self.jsonl_path = self.log_dir / jsonl_name
        self._fieldnames: List[str] = []

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        stdout: bool = True,
    ) -> Dict[str, object]:
        row = dict(metrics)
        if prefix:
            row = {f"{prefix}/{key}": value for key, value in row.items()}
        if step is not None:
            row["step"] = int(step)
        row["timestamp"] = time.time()

        self._append_jsonl(row)
        self._append_csv(row)
        if stdout:
            self._print_row(row)
        return row

    def save_config(self, config: Mapping[str, object], filename: str = "config.json") -> Path:
        path = self.log_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        return path

    def log_text(self, lines: str | Iterable[str], filename: str = "notes.txt") -> Path:
        path = self.log_dir / filename
        with path.open("a", encoding="utf-8") as f:
            if isinstance(lines, str):
                f.write(lines.rstrip() + "\n")
            else:
                for line in lines:
                    f.write(str(line).rstrip() + "\n")
        return path

    def _append_jsonl(self, row: Mapping[str, object]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _append_csv(self, row: Mapping[str, object]) -> None:
        row_keys = list(row.keys())
        if not self._fieldnames:
            self._fieldnames = row_keys
            self._write_csv_rows([row], rewrite=True)
            return

        missing_keys = [key for key in row_keys if key not in self._fieldnames]
        if missing_keys:
            self._fieldnames.extend(missing_keys)
            existing_rows = self._read_csv_rows()
            existing_rows.append(dict(row))
            self._write_csv_rows(existing_rows, rewrite=True)
            return

        self._write_csv_rows([row], rewrite=False)

    def _read_csv_rows(self) -> List[Dict[str, object]]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def _write_csv_rows(self, rows: Iterable[Mapping[str, object]], rewrite: bool) -> None:
        mode = "w" if rewrite else "a"
        with self.csv_path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if rewrite or f.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in self._fieldnames})

    def _print_row(self, row: Mapping[str, object]) -> None:
        parts = []
        for key, value in row.items():
            if key == "timestamp":
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        print(" | ".join(parts))
