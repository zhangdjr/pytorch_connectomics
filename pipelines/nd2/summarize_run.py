#!/usr/bin/env python3
"""
Summarize an ND2 batch run.

Outputs:
  - <run_root>/case_status_and_durations.tsv
  - <run_root>/run_summary.md
  - <run_root>/run_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


START_RE = re.compile(r"^Start:\s+(.*)$")
DONE_RE = re.compile(r"^ALL DONE:\s+(.*)$")


@dataclass
class ParsedError:
    kind: str
    detail: str


def parse_log_datetime(raw: str) -> Optional[datetime]:
    text = raw.strip()
    if not text:
        return None

    candidates = [text]
    parts = text.split()
    # Convert "Sun Mar 29 10:03:10 EDT 2026" -> "Sun Mar 29 10:03:10 2026"
    if len(parts) == 6:
        candidates.insert(0, " ".join(parts[:4] + [parts[5]]))

    for cand in candidates:
        for fmt in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(cand, fmt)
            except ValueError:
                continue
    return None


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="ignore").splitlines()
    except Exception:
        return []


def parse_error_from_files(paths: Iterable[Path]) -> ParsedError:
    known_patterns = [
        ("no_cuda_gpu", "RuntimeError: No CUDA GPUs are available"),
        ("missing_nvidia_smi", "ERROR: nvidia-smi not found"),
        ("nvidia_smi_failed", "ERROR: nvidia-smi failed"),
        ("no_gpu_reported", "ERROR: No GPUs reported by nvidia-smi"),
        ("torch_cuda_unavailable", "ERROR: torch.cuda.is_available() == False"),
        ("nd2_no_position_axis", "KeyError: 'P'"),
    ]

    for path in paths:
        text = path.read_text(errors="ignore") if path.exists() else ""
        if not text:
            continue

        for kind, marker in known_patterns:
            if marker in text:
                return ParsedError(kind=kind, detail=marker)

        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                for tail in reversed(lines[i + 1 :]):
                    tail = tail.strip()
                    if tail:
                        return ParsedError(kind="traceback", detail=tail)
                return ParsedError(kind="traceback", detail="Traceback without final error line")

        for line in lines:
            line = line.strip()
            if line.startswith("ERROR:"):
                return ParsedError(kind="error_line", detail=line)

    return ParsedError(kind="", detail="")


def get_case_ids(run_root: Path) -> list[str]:
    manifest = run_root / "manifest.csv"
    if manifest.exists():
        ids = []
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = row.get("nd2_id", "").strip()
                if case:
                    ids.append(case)
        # Keep order while de-duplicating
        seen = set()
        ordered = []
        for c in ids:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    return sorted([p.name for p in run_root.iterdir() if p.is_dir() and p.name.startswith("1-")])


def summarize_case(run_root: Path, case_id: str) -> dict:
    case_dir = run_root / case_id
    logs_dir = case_dir / "logs"
    pred_dir = case_dir / "pred"
    postproc_dir = case_dir / "postproc"
    qc_path = case_dir / "qc" / "check_report.json"
    tile_names_path = case_dir / "meta" / "tile_names.txt"

    extract_out = sorted(logs_dir.glob("nd2_extract_*.out")) if logs_dir.exists() else []
    extract_err = sorted(logs_dir.glob("nd2_extract_*.err")) if logs_dir.exists() else []
    infer_out = sorted(logs_dir.glob("nd2_infer_*.out")) if logs_dir.exists() else []
    infer_err = sorted(logs_dir.glob("nd2_infer_*.err")) if logs_dir.exists() else []
    post_out = sorted(logs_dir.glob("nd2_postproc_*.out")) if logs_dir.exists() else []
    post_err = sorted(logs_dir.glob("nd2_postproc_*.err")) if logs_dir.exists() else []
    all_logs = sorted(logs_dir.glob("*")) if logs_dir.exists() else []

    start_times = []
    for p in extract_out:
        for line in read_lines(p):
            m = START_RE.match(line)
            if m:
                dt = parse_log_datetime(m.group(1))
                if dt is not None:
                    start_times.append(dt)
    start_dt = min(start_times) if start_times else None

    end_times = []
    for p in post_out:
        for line in read_lines(p):
            m = DONE_RE.match(line)
            if m:
                dt = parse_log_datetime(m.group(1))
                if dt is not None:
                    end_times.append(dt)
    end_dt = max(end_times) if end_times else None

    if end_dt is None and all_logs:
        end_dt = datetime.fromtimestamp(max(p.stat().st_mtime for p in all_logs))
    if start_dt is None and all_logs:
        start_dt = datetime.fromtimestamp(min(p.stat().st_mtime for p in all_logs))

    duration_sec = None
    if start_dt is not None and end_dt is not None and end_dt >= start_dt:
        duration_sec = int((end_dt - start_dt).total_seconds())

    planned_tiles = 0
    if tile_names_path.exists():
        planned_tiles = len([x for x in tile_names_path.read_text().splitlines() if x.strip()])

    pred_tiff_count = len(list(pred_dir.glob("*_ch1_prediction.tiff"))) if pred_dir.exists() else 0
    tile_csv_count = len(list(postproc_dir.glob("*_fiber_coordinates.csv"))) if postproc_dir.exists() else 0
    master_csv_exists = (postproc_dir / "all_tiles_fiber_coordinates.csv").exists()

    qc_exists = qc_path.exists()
    if qc_exists:
        try:
            qc = json.loads(qc_path.read_text())
            pred_tiff_count = int(qc.get("prediction_tiff_count", pred_tiff_count) or 0)
            tile_csv_count = int(qc.get("tile_csv_count", tile_csv_count) or 0)
            master_csv_exists = bool(qc.get("master_csv_exists", master_csv_exists))
        except Exception:
            pass

    ok = qc_exists and master_csv_exists and pred_tiff_count > 0 and tile_csv_count > 0

    if ok:
        stage = "success"
    elif extract_err and not infer_out and not infer_err:
        stage = "extract_failed"
    elif infer_out or infer_err:
        stage = "infer_failed" if not post_out and not post_err else "postproc_failed"
    elif post_out or post_err:
        stage = "postproc_failed"
    else:
        stage = "failed_unknown"

    if stage == "extract_failed":
        parsed_err = parse_error_from_files(extract_err + extract_out)
    elif stage == "infer_failed":
        parsed_err = parse_error_from_files(infer_err + extract_err)
    elif stage == "postproc_failed":
        parsed_err = parse_error_from_files(post_err + infer_err + extract_err)
    else:
        parsed_err = ParsedError(kind="", detail="")

    def fmt_dt(dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""

    def fmt_hms(sec: Optional[int]) -> str:
        if sec is None:
            return ""
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    return {
        "case": case_id,
        "ok": ok,
        "status_stage": stage,
        "start": fmt_dt(start_dt),
        "end": fmt_dt(end_dt),
        "duration_hms": fmt_hms(duration_sec),
        "duration_sec": "" if duration_sec is None else duration_sec,
        "planned_tiles": planned_tiles,
        "prediction_tiff_count": pred_tiff_count,
        "tile_csv_count": tile_csv_count,
        "master_csv_exists": master_csv_exists,
        "qc_exists": qc_exists,
        "error_kind": parsed_err.kind,
        "error_detail": parsed_err.detail.replace("\t", " ").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ND2 pipeline run")
    parser.add_argument("--run-root", required=True, help="Run directory path")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    case_ids = get_case_ids(run_root)
    rows = [summarize_case(run_root, case_id) for case_id in case_ids]
    rows.sort(key=lambda r: r["case"])

    tsv_path = run_root / "case_status_and_durations.tsv"
    md_path = run_root / "run_summary.md"
    json_path = run_root / "run_summary.json"

    if rows:
        with tsv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    else:
        tsv_path.write_text("")

    total_cases = len(rows)
    success_cases = sum(1 for r in rows if r["ok"])
    failed_cases = total_cases - success_cases
    all_secs = [int(r["duration_sec"]) for r in rows if str(r["duration_sec"]).isdigit()]
    ok_secs = [int(r["duration_sec"]) for r in rows if r["ok"] and str(r["duration_sec"]).isdigit()]

    sum_all = sum(all_secs)
    sum_ok = sum(ok_secs)

    valid_rows = [r for r in rows if r["start"] and r["end"]]
    if valid_rows:
        min_start = min(datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S") for r in valid_rows)
        max_end = max(datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S") for r in valid_rows)
        wall_sec = int((max_end - min_start).total_seconds())
        earliest_start = min_start.strftime("%Y-%m-%d %H:%M:%S")
        latest_end = max_end.strftime("%Y-%m-%d %H:%M:%S")
    else:
        wall_sec = None
        earliest_start = ""
        latest_end = ""

    def fmt_hms(sec: Optional[int]) -> str:
        if sec is None:
            return ""
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    failed_rows = [r for r in rows if not r["ok"]]

    md_lines = [
        f"# ND2 Run Summary: {run_root.name}",
        "",
        f"- Run root: `{run_root}`",
        f"- Total cases: **{total_cases}**",
        f"- Success: **{success_cases}**",
        f"- Failed: **{failed_cases}**",
        f"- Sum duration (all cases): **{fmt_hms(sum_all)}**",
        f"- Sum duration (success cases): **{fmt_hms(sum_ok)}**",
        f"- Wall span: **{fmt_hms(wall_sec)}**",
        f"- Earliest start: `{earliest_start}`",
        f"- Latest end: `{latest_end}`",
        "",
        "## Failed Cases",
        "",
    ]

    if not failed_rows:
        md_lines.append("- None")
    else:
        md_lines.append("| case | stage | duration | error_kind | error_detail |")
        md_lines.append("|---|---|---:|---|---|")
        for row in failed_rows:
            detail = row["error_detail"].replace("|", "/")
            if len(detail) > 180:
                detail = detail[:177] + "..."
            md_lines.append(
                f"| {row['case']} | {row['status_stage']} | {row['duration_hms']} | "
                f"{row['error_kind']} | {detail} |"
            )

    md_lines.extend(["", "## Per Case", "", "| case | ok | duration | planned | pred_tiff | tile_csv |", "|---|---|---:|---:|---:|---:|"])
    for row in rows:
        md_lines.append(
            f"| {row['case']} | {row['ok']} | {row['duration_hms']} | "
            f"{row['planned_tiles']} | {row['prediction_tiff_count']} | {row['tile_csv_count']} |"
        )
    md_lines.append("")

    md_path.write_text("\n".join(md_lines))

    json_path.write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "total_cases": total_cases,
                "success_cases": success_cases,
                "failed_cases": failed_cases,
                "sum_duration_all_hms": fmt_hms(sum_all),
                "sum_duration_success_hms": fmt_hms(sum_ok),
                "wall_span_hms": fmt_hms(wall_sec),
                "earliest_start": earliest_start,
                "latest_end": latest_end,
                "failed_case_ids": [r["case"] for r in failed_rows],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    print(f"Run root: {run_root}")
    print(f"Total: {total_cases}, Success: {success_cases}, Failed: {failed_cases}")
    print(f"Summary markdown: {md_path}")
    print(f"Summary table:    {tsv_path}")
    print(f"Summary json:     {json_path}")


if __name__ == "__main__":
    main()
