#!/usr/bin/env python3
"""Summarise Clang -ftime-trace JSON files.

Usage:
    python3 summarize_traces.py <trace_dir> [--top N] [--min-ms M] [--category CAT]

Categories of interest:
    InstantiateClass, InstantiateFunction, ParseTemplate,
    PerformPendingInstantiations, Source, Frontend, Backend, Total
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_traces(trace_dir: Path) -> list[dict]:
    events = []
    for f in sorted(trace_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  [skip] {f.name}: invalid JSON", file=sys.stderr)
            continue
        for ev in data.get("traceEvents", []):
            if ev.get("ph") == "X" and ev.get("dur", 0) > 0:
                ev["_file"] = f.name
                events.append(ev)
    return events


def summarise(events: list[dict], top: int, min_ms: float, category: str | None):
    by_cat: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    for ev in events:
        name = ev.get("name", "")
        dur_ms = ev["dur"] / 1000.0
        if dur_ms < min_ms:
            continue
        if category and name != category:
            continue
        detail = ev.get("args", {}).get("detail", "")
        by_cat[name].append((dur_ms, detail, ev["_file"]))

    cats_by_total = sorted(
        by_cat.items(),
        key=lambda kv: sum(t for t, _, _ in kv[1]),
        reverse=True,
    )

    for cat, entries in cats_by_total:
        entries.sort(reverse=True)
        total_ms = sum(t for t, _, _ in entries)
        print(f"\n{'=' * 70}")
        print(f"  {cat}  —  {len(entries)} events  —  total {total_ms:.1f} ms")
        print(f"{'=' * 70}")
        for dur, detail, fname in entries[:top]:
            label = detail[:80] if detail else "(no detail)"
            print(f"  {dur:8.1f} ms   {label}   [{fname}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--top", type=int, default=20, help="entries per category")
    ap.add_argument("--min-ms", type=float, default=1.0, help="ignore events shorter than N ms")
    ap.add_argument("--category", type=str, default=None, help="filter to one category name")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        sys.exit(f"Not a directory: {args.trace_dir}")

    events = load_traces(args.trace_dir)
    if not events:
        sys.exit(f"No trace events found in {args.trace_dir}")

    print(f"Loaded {len(events)} events from {args.trace_dir}")
    summarise(events, args.top, args.min_ms, args.category)


if __name__ == "__main__":
    main()
