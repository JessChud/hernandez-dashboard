#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = "jchud-stanford-university/hernandez-replication"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh dashboard rows currently marked running from exact W&B run IDs."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--runs-csv", type=Path, default=Path("runs.csv"))
    parser.add_argument("--index-html", type=Path, default=Path("index.html"))
    parser.add_argument("--query-json", type=Path, default=Path("/tmp/hernandez_running_exact_latest.json"))
    parser.add_argument(
        "--include-run-id",
        action="append",
        default=[],
        help="Also query and add this W&B run ID if it is not already in runs.csv.",
    )
    return parser.parse_args()


def run_id_from_link(link: str | None) -> str:
    return str(link or "").rstrip("/").split("/")[-1]


def load_running_rows(runs_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = list(csv.DictReader(runs_csv.open(newline="")))
    run_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("State", "")).lower() != "running":
            continue
        run_id = run_id_from_link(row.get("W&B Link"))
        if run_id and run_id not in seen:
            run_ids.append(run_id)
            seen.add(run_id)
    return rows, run_ids


def query_wandb(project: str, rows: list[dict[str, str]], run_ids: list[str]) -> dict[str, Any]:
    import wandb

    row_by_run_id = {run_id_from_link(row.get("W&B Link")): row for row in rows}
    api = wandb.Api(timeout=60)
    records: list[dict[str, Any]] = []
    for run_id in run_ids:
        row = row_by_run_id.get(run_id, {})
        record: dict[str, Any] = {"id": run_id, "row": row}
        try:
            run = api.run(f"{project}/{run_id}")
            summary = dict(run.summary._json_dict)
            record.update(
                {
                    "state": run.state,
                    "name": run.name,
                    "config": dict(run.config),
                    "created_at": str(run.created_at),
                    "sweep": getattr(run.sweep, "id", None) if run.sweep else None,
                    "runtime": summary.get("_runtime") or summary.get("runtime"),
                    "step": summary.get("train/global_step")
                    or summary.get("trainer/global_step")
                    or summary.get("global_step")
                    or summary.get("_step"),
                    "epoch": summary.get("epoch")
                    or summary.get("train/epoch")
                    or summary.get("trainer/epoch"),
                    "train_loss": summary.get("train/loss")
                    or summary.get("train_loss")
                    or summary.get("loss/train")
                    or summary.get("loss"),
                    "eval_loss": summary.get("eval/loss")
                    or summary.get("eval_loss")
                    or summary.get("test/loss")
                    or summary.get("loss/eval"),
                    "tokens_seen": summary.get("tokens_seen")
                    or summary.get("tokens/seen")
                    or summary.get("total_tokens"),
                    "tok_s": summary.get("train/tokens_per_second")
                    or summary.get("tokens_per_second"),
                    "url": run.url,
                }
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = repr(exc)
        records.append(record)
    return {"queried_at": dt.datetime.now(dt.timezone.utc).isoformat(), "runs": records}


def nested_value(config: dict[str, Any], section: str, key: str, default: Any = "") -> Any:
    value = config.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def model_label(model_name: Any) -> str:
    text = str(model_name or "")
    match = re.search(r"Qwen3[-/](\d+M)", text, re.I)
    if match:
        return match.group(1)
    return text.rsplit("/", 1)[-1] if text else ""


def rep_budget_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    percent = numeric * 100
    if percent.is_integer():
        return f"{int(percent)}%"
    return f"{percent:g}%"


def created_at_for_row(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_from_record(record: dict[str, Any], fieldnames: list[str]) -> dict[str, str]:
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    data_config = config.get("data_config", {}) if isinstance(config.get("data_config"), dict) else {}
    trainer_config = config.get("trainer_config", {}) if isinstance(config.get("trainer_config"), dict) else {}
    model_config = config.get("model_config", {}) if isinstance(config.get("model_config"), dict) else {}

    num_repeats = data_config.get("num_repeats", nested_value(config, "data_config", "num_repeats"))
    repetition_budget = data_config.get(
        "repetition_budget", nested_value(config, "data_config", "repetition_budget")
    )
    model = model_label(model_config.get("model_name", nested_value(config, "model_config", "model_name")))
    direction = str(data_config.get("direction", nested_value(config, "data_config", "direction")) or "")
    overtrain = trainer_config.get(
        "overtrain_multiplier", nested_value(config, "trainer_config", "overtrain_multiplier", 1)
    )

    row = {field: "" for field in fieldnames}
    row.update(
        {
            "Name": str(record.get("name") or ""),
            "State": str(record.get("state") or ""),
            "Sweep ID": str(record.get("sweep") or ""),
            "Sweep Label": f"{model} nr=[{num_repeats}]" if model and num_repeats != "" else "",
            "Model": model,
            "Repeated": "non-repeated" if str(num_repeats) in {"", "1", "1.0"} else "repeated",
            "Num Repeats": fmt_num(num_repeats, 0),
            "Rep Budget": rep_budget_label(repetition_budget),
            "Direction": direction,
            "OT Multiplier": fmt_num(overtrain, 2) or "1",
            "Created": created_at_for_row(record.get("created_at")),
            "W&B Link": str(record.get("url") or ""),
        }
    )
    return row


def add_included_rows(
    rows: list[dict[str, str]], records: dict[str, dict[str, Any]], include_run_ids: list[str]
) -> list[str]:
    if not include_run_ids:
        return []
    fieldnames = list(rows[0]) if rows else []
    existing = {run_id_from_link(row.get("W&B Link")) for row in rows}
    added: list[str] = []
    for run_id in include_run_ids:
        if run_id in existing:
            continue
        record = records.get(run_id)
        if not record:
            continue
        rows.append(row_from_record(record, fieldnames))
        existing.add(run_id)
        added.append(run_id)
    return added


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000 and number.is_integer():
        return str(int(number))
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def human_duration(seconds: Any, running: bool) -> str:
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return ""
    minutes = int(total_seconds // 60)
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        text = f"{hours}h {mins}m"
    elif hours:
        text = f"{hours}h"
    else:
        text = f"{mins}m"
    return f"{text} (running)" if running else text


def update_csv(runs_csv: Path, rows: list[dict[str, str]], records: dict[str, dict[str, Any]]) -> list[tuple[Any, ...]]:
    changed: list[tuple[Any, ...]] = []
    fieldnames = list(rows[0]) if rows else []
    for row in rows:
        run_id = run_id_from_link(row.get("W&B Link"))
        record = records.get(run_id)
        if not record:
            continue
        before = (
            row.get("State"),
            row.get("Train Loss (last)"),
            row.get("Global Step"),
            row.get("Epoch"),
            row.get("Duration"),
        )
        row["State"] = str(record.get("state") or row.get("State") or "")
        if record.get("name"):
            row["Name"] = str(record["name"])
        if record.get("sweep"):
            row["Sweep ID"] = str(record["sweep"])
        if record.get("train_loss") is not None:
            row["Train Loss (last)"] = fmt_num(record.get("train_loss"), 4)
        if record.get("eval_loss") is not None:
            row["Eval Loss"] = fmt_num(record.get("eval_loss"), 4)
        if record.get("step") is not None:
            row["Global Step"] = str(int(float(record["step"])))
        if record.get("epoch") is not None:
            row["Epoch"] = str(record["epoch"])
        elif str(record.get("state", "")).lower() == "running":
            row["Epoch"] = ""
        if record.get("tokens_seen") is not None:
            row["Tokens Seen"] = str(int(float(record["tokens_seen"])))
        if record.get("tok_s") is not None:
            row["Train Tok/s"] = str(record["tok_s"])
        if record.get("runtime") is not None:
            row["Duration"] = human_duration(record["runtime"], str(record.get("state", "")).lower() == "running")
        after = (
            row.get("State"),
            row.get("Train Loss (last)"),
            row.get("Global Step"),
            row.get("Epoch"),
            row.get("Duration"),
        )
        if before != after:
            changed.append((run_id, row.get("Model"), row.get("Rep Budget"), row.get("Num Repeats"), before, after))

    with runs_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def human_tokens(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1e9:
        return f"{number / 1e9:.2f}B"
    if abs(number) >= 1e6:
        return f"{number / 1e6:.1f}M"
    if abs(number) >= 1e3:
        return f"{number / 1e3:.1f}K"
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def fmt_tps(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def td(content: str) -> str:
    return f"  <td>{content}</td>"


def link_cell(url: str | None, label: str | None) -> str:
    return td(f'<a href="{html.escape(url or "#")}" target="_blank">{html.escape(label or "")}</a>')


def state_cell(state: str | None) -> str:
    color = {
        "running": "#22c55e",
        "finished": "#6b7280",
        "failed": "#ef4444",
        "crashed": "#ef4444",
    }.get(str(state or "").lower(), "#f59e0b")
    return td(
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px">{html.escape(state or "")}</span>'
    )


def rebuild_html_row(row: dict[str, str]) -> str:
    sweep_id = row.get("Sweep ID") or ""
    sweep_url = (
        f"https://wandb.ai/jchud-stanford-university/hernandez-replication/sweeps/{sweep_id}"
        if sweep_id
        else "#"
    )
    cells = [
        link_cell(row.get("W&B Link"), row.get("Name")),
        state_cell(row.get("State")),
        link_cell(sweep_url, sweep_id),
        td(html.escape(row.get("Model") or "")),
        td(html.escape(row.get("Num Repeats") or "")),
        td(html.escape(row.get("Direction") or "")),
        td(f"{html.escape(row.get('OT Multiplier') or '')}x" if row.get("OT Multiplier") else ""),
        td(html.escape(row.get("Eval Loss") or "-")),
        td(html.escape(row.get("Train Loss (last)") or "")),
        td(html.escape(row.get("Global Step") or "")),
        td(html.escape(fmt_float(row.get("Epoch"), 3))),
        td(html.escape(human_tokens(row.get("Tokens Seen")))),
        td(html.escape(fmt_tps(row.get("Train Tok/s")))),
        td(html.escape(row.get("Duration") or "")),
        td(html.escape(row.get("Finished") or "")),
        td(html.escape(row.get("Created") or "")),
    ]
    return "<tr>\n" + "\n".join(cells) + "\n</tr>"


def current_build_time(index_text: str) -> dt.datetime:
    match = re.search(r'DASHBOARD_BUILD_VERSION = "(\d{14})UTC"', index_text)
    if not match:
        return dt.datetime.now(dt.timezone.utc)
    return dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def update_stat(text: str, label: str, value: int) -> str:
    pattern = re.compile(
        r'(<div class="stat"><div class="stat-num">)\d+(</div><div class="stat-label">'
        + re.escape(label)
        + r"</div></div>)"
    )
    text, count = pattern.subn(rf"\g<1>{value}\2", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update top stat {label!r}")
    return text


def update_top_stats(text: str, rows: list[dict[str, str]]) -> str:
    state_counts = Counter(str(row.get("State", "")) for row in rows)
    text = update_stat(text, "Running", state_counts.get("running", 0))
    text = update_stat(text, "Finished", state_counts.get("finished", 0))
    text = update_stat(text, "Total Runs", len(rows))
    return text


def replace_table_row(text: str, run_id: str, row_html: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"<tr>\s*(?:(?!</tr>).)*?"
        + re.escape(f"/runs/{run_id}")
        + r"(?:(?!</tr>).)*?</tr>",
        re.S,
    )
    text, count = pattern.subn(row_html, text, count=1)
    return text, count == 1


def update_index(
    index_html: Path,
    rows: list[dict[str, str]],
    records: dict[str, dict[str, Any]],
    queried_at: str,
) -> str:
    text = index_html.read_text()
    query_time = dt.datetime.fromisoformat(queried_at).astimezone(dt.timezone.utc)
    build_time = max(query_time, current_build_time(text) + dt.timedelta(seconds=1))
    build = build_time.strftime("%Y%m%d%H%M%SUTC")
    updated = build_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    text = re.sub(
        r'Updated: [0-9:\- ]+ UTC <span style="color:#9ca3af">\| artifact [0-9A-Z]+; CSV cache-busted</span>',
        f'Updated: {updated} <span style="color:#9ca3af">| artifact {build}; CSV cache-busted</span>',
        text,
        count=1,
    )
    text = re.sub(r'DASHBOARD_BUILD_VERSION = "[0-9A-Z]+"', f'DASHBOARD_BUILD_VERSION = "{build}"', text, count=1)
    text = update_top_stats(text, rows)

    row_by_run_id = {run_id_from_link(row.get("W&B Link")): row for row in rows}
    missing_rows: list[str] = []
    for run_id in records:
        row = row_by_run_id.get(run_id)
        if not row:
            continue
        text, replaced = replace_table_row(text, run_id, rebuild_html_row(row))
        if not replaced:
            missing_rows.append(run_id)
    if missing_rows:
        raise RuntimeError(f"Could not update table rows for run ids: {', '.join(missing_rows)}")
    index_html.write_text(text)
    return build


def main() -> None:
    args = parse_args()
    rows, run_ids = load_running_rows(args.runs_csv)
    seen_run_ids = set(run_ids)
    for run_id in args.include_run_id:
        if run_id not in seen_run_ids:
            run_ids.append(run_id)
            seen_run_ids.add(run_id)
    query = query_wandb(args.project, rows, run_ids)
    args.query_json.write_text(json.dumps(query, indent=2, sort_keys=True) + "\n")
    records = {record["id"]: record for record in query["runs"] if "error" not in record}
    added = add_included_rows(rows, records, args.include_run_id)
    changed = update_csv(args.runs_csv, rows, records)
    build = update_index(args.index_html, rows, records, str(query["queried_at"]))

    print(f"build {build}")
    print(f"queried {len(query['runs'])}")
    print(f"added {len(added)}")
    for run_id in added:
        print(f"added {run_id}")
    print(f"changed {len(changed)}")
    for run_id, model, rep_budget, num_repeats, before, after in changed:
        print(f"{run_id} {model} {rep_budget} nr={num_repeats} {before} -> {after}")
    errors = [record for record in query["runs"] if "error" in record]
    if errors:
        print(f"errors {len(errors)}")
        for record in errors:
            print(f"{record['id']} {record['error']}")


if __name__ == "__main__":
    main()
