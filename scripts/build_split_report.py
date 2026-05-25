#!/usr/bin/env python3
"""Build reports/split_report.md from converter artifacts.

Pure reader — does not recompute any split logic.

Usage:
    python scripts/build_split_report.py --out-dir data/out --report reports/split_report.md
"""
import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{100 * n // total}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def _dim_table(overall: dict, by_split: dict, dim: str) -> str:
    all_keys = sorted(set(overall.get(dim, {}).keys()))
    if not all_keys:
        return "_no data_"
    headers = [dim, "overall", "train", "dev", "test"]
    total_overall = sum(overall.get(dim, {}).values())
    rows = []
    for k in all_keys:
        ov = overall.get(dim, {}).get(k, 0)
        tr = by_split.get("train", {}).get(dim, {}).get(k, 0)
        dv = by_split.get("dev", {}).get(dim, {}).get(k, 0)
        te = by_split.get("test", {}).get(dim, {}).get(k, 0)
        rows.append([k, f"{ov} ({_pct(ov, total_overall)})", str(tr), str(dv), str(te)])
    return _table(headers, rows)


def _check_warnings(dist: dict, manifest: dict) -> list[str]:
    warnings = []
    overall = dist.get("overall", {})
    by_split = dist.get("by_split", {})

    all_scenario_types = set(overall.get("scenario_type", {}).keys())
    for split_name in ("train", "dev", "test"):
        split_scenarios = set(by_split.get(split_name, {}).get("scenario_type", {}).keys())
        missing = all_scenario_types - split_scenarios
        for s in sorted(missing):
            warnings.append(f"WARNING: `{split_name}` split has zero simulations for scenario_type `{s}` (present in overall)")

    for split_name in ("dev", "test"):
        hard_count = by_split.get(split_name, {}).get("difficulty_bucket", {}).get("hard", 0)
        if hard_count == 0:
            warnings.append(f"WARNING: `{split_name}` split has zero `hard` difficulty simulations")

    train_fingerprints = set(by_split.get("train", {}).get("terminal_tool_fingerprint", {}).keys())
    dev_fingerprints = set(by_split.get("dev", {}).get("terminal_tool_fingerprint", {}).keys())
    test_fingerprints = set(by_split.get("test", {}).get("terminal_tool_fingerprint", {}).keys())
    train_only = train_fingerprints - dev_fingerprints - test_fingerprints
    for fp in sorted(train_only):
        if fp != "none":
            warnings.append(f"WARNING: terminal_tool_fingerprint `{fp}` appears only in train (not in dev or test)")

    total_overall = sum(overall.get("requires_grounding", {}).values()) or 1
    overall_grounding_rate = overall.get("requires_grounding", {}).get("True", 0) / total_overall
    for split_name in ("dev", "test"):
        split_total = sum(by_split.get(split_name, {}).get("requires_grounding", {}).values()) or 1
        split_rate = by_split.get(split_name, {}).get("requires_grounding", {}).get("True", 0) / split_total
        diff_pct = abs(split_rate - overall_grounding_rate) * 100
        if diff_pct > 15:
            warnings.append(
                f"WARNING: `{split_name}` grounding rate {split_rate:.0%} differs from overall "
                f"{overall_grounding_rate:.0%} by {diff_pct:.0f}pp (> 15pp threshold)"
            )

    return warnings


def build_report(out_dir: Path, report_path: Path) -> str:
    dist = _load_json(out_dir / "scenario_distribution.json")
    manifest = _load_json(out_dir / "split_manifest.json")
    profiles = _load_jsonl(out_dir / "simulation_profiles.jsonl")

    splits_dir = out_dir / "splits"
    train_ids = _load_json(splits_dir / "train_simulation_ids.json")
    dev_ids = _load_json(splits_dir / "dev_simulation_ids.json")
    test_ids = _load_json(splits_dir / "test_simulation_ids.json")

    overall = dist.get("overall", {})
    by_split = dist.get("by_split", {})

    warnings = _check_warnings(dist, manifest)

    lines = [
        "# Split Report",
        "",
        "> **Descriptive split validation artifact.**  ",
        "> Not an input to graph recommendation.  ",
        "> `experimental_boundary.artifact_type = descriptive` | `data_scope = all_splits` | `allowed_to_influence_graph = false`",
        "",
        f"**Version:** `{manifest.get('split_version', 'n/a')}`  ",
        f"**Seed:** {manifest.get('seed', 'n/a')}  ",
        f"**Strategy:** primary = {manifest.get('strategy', {}).get('primary', [])}  ",
        f"**Ratio:** train {manifest['ratio']['train']} / dev {manifest['ratio']['dev']} / test {manifest['ratio']['test']}",
        "",
        "## Split Sizes",
        "",
        _table(
            ["split", "simulations", "turns"],
            [
                ["train", str(len(train_ids)), str(sum(1 for p in profiles if p["split"] == "train"))],
                ["dev",   str(len(dev_ids)),   str(sum(1 for p in profiles if p["split"] == "dev"))],
                ["test",  str(len(test_ids)),  str(sum(1 for p in profiles if p["split"] == "test"))],
                ["total", str(len(profiles)),  str(len(profiles))],
            ],
        ),
        "",
    ]

    dims_to_show = [
        ("scenario_type", "Scenario Type"),
        ("primary_scenario", "Primary Scenario"),
        ("difficulty_bucket", "Difficulty Bucket"),
        ("requires_grounding", "Requires Grounding"),
        ("has_item_ids", "Has item_ids / new_item_ids"),
        ("has_order_id", "Has order_id"),
        ("has_product_id", "Has product_id"),
        ("requires_tool_chaining", "Requires Tool Chaining"),
        ("is_multi_action", "Is Multi-Action"),
        ("terminal_tool_fingerprint", "Terminal Tool Fingerprint"),
    ]

    for dim_key, dim_label in dims_to_show:
        lines += [f"## {dim_label}", "", _dim_table(overall, by_split, dim_key), ""]

    if warnings:
        lines += ["## Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines += ["## Warnings", "", "_No warnings — split looks healthy._", ""]

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split_report.md from converter artifacts")
    parser.add_argument("--out-dir", default="data/out", help="Converter output directory")
    parser.add_argument("--report", default="reports/split_report.md", help="Output report path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    report = build_report(out_dir, report_path)

    print(f"Written: {report_path}  ({report_path.stat().st_size:,} bytes)")

    lines = report.splitlines()
    warning_lines = [l for l in lines if l.startswith("- WARNING:")]
    if warning_lines:
        print(f"\n{len(warning_lines)} warning(s):")
        for w in warning_lines:
            print(f"  {w}")
    else:
        print("No warnings — split looks healthy.")


if __name__ == "__main__":
    main()
