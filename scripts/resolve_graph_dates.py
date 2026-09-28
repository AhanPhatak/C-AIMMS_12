"""Migrate cached CTC graphs so relative dates are resolved in the memory text.

Writes a NEW cache directory; the source cache is only read. Per episodic
content node it rewrites exactly two fields and nothing else -- cue and tag
links, ids, layers and semantic nodes are copied byte-for-byte, so retrieval
still walks the same graph and no LLM call is needed:

  time  '1:56 pm on 8 May, 2023'  ->  '8 May, 2023'
        (date only, in the timestamp's own style; --keep-time-of-day to skip)
  text  '... support group yesterday and ...'
        -> '... support group yesterday (7 May, 2023) and ...'
        (iterret.time_resolution; --no-resolve to skip)

--fix-doubled-prefix additionally strips the '[time] Speaker: ' copy that
graphs built before the fork port carry inside their text. It is off by
default so its effect can be measured separately from the date change.

Since 2026-09-28 this is OPTIONAL: CueTagContentGraph resolves relative dates
itself whenever a graph is built or loaded (ITERRET_RESOLVE_DATES, default on),
giving byte-identical text to this script's default output. It remains useful to
write resolved files to disk, and for --fix-doubled-prefix. Point the eval at a
migrated cache with WORKMEM_GRAPH_CACHE_DIR=<dst>.

  source env.sh
  python3 scripts/resolve_graph_dates.py                    # graph_cache -> graph_cache_dates
  python3 scripts/resolve_graph_dates.py --dst X --fix-doubled-prefix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IterRet"))
from iterret.time_resolution import date_only, resolve_relative_time  # noqa: E402

_OUT = os.environ.get("CAIMMS_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs"))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def migrate_graph(graph: dict, *, resolve: bool, keep_time_of_day: bool,
                  fix_doubled_prefix: bool, stats: Counter) -> dict:
    for node in graph["contents"].values():
        original_time = node.get("time")
        if node.get("layer") != "episodic" or not original_time:
            continue
        stats["episodic_nodes"] += 1
        text = node["text"]
        if fix_doubled_prefix:
            # "Caroline: [<time>] Caroline: Hey" -> "Caroline: Hey"
            m = re.match(rf"([^:\n]+): \[{re.escape(original_time)}\] \1: ", text)
            if m:
                text = f"{m.group(1)}: {text[m.end():]}"
                stats["doubled_prefix_removed"] += 1
        if resolve:
            text, n = resolve_relative_time(text, original_time)
            if n:
                stats["nodes_with_resolution"] += 1
                stats["expressions_resolved"] += n
        if not keep_time_of_day:
            new_time = date_only(original_time)
            if new_time != original_time:
                stats["times_shortened"] += 1
            node["time"] = new_time
        node["text"] = text
    return graph


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{_OUT}/graph_cache")
    ap.add_argument("--dst", default=f"{_OUT}/graph_cache_dates")
    ap.add_argument("--no-resolve", action="store_true", help="do not annotate relative dates")
    ap.add_argument("--keep-time-of-day", action="store_true", help="leave node.time unchanged")
    ap.add_argument("--fix-doubled-prefix", action="store_true", help="strip the duplicated '[time] Speaker: '")
    ap.add_argument("--force", action="store_true", help="overwrite files already in --dst")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if src == dst:
        sys.exit("--dst must differ from --src: the source cache is never modified.")
    files = sorted(src.glob("sample_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        sys.exit(f"No sample_*.json in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    clash = [f.name for f in files if (dst / f.name).exists()]
    if clash and not args.force:
        sys.exit(f"{dst} already has {clash}; pass --force to overwrite.")

    flags = {"resolve": not args.no_resolve, "keep_time_of_day": args.keep_time_of_day,
             "fix_doubled_prefix": args.fix_doubled_prefix}
    manifest = {"source": str(src), "flags": flags, "samples": {}}
    total = Counter()
    for f in files:
        stats = Counter()
        graph = migrate_graph(json.loads(f.read_text()), stats=stats, **flags)
        out = dst / f.name
        out.write_text(json.dumps(graph, indent=2))
        manifest["samples"][f.name] = {"source_md5": _md5(f), "output_md5": _md5(out), **stats}
        total.update(stats)
        print(f"{f.name}: {dict(stats)}")
    (dst / "MIGRATION.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(files)} graphs -> {dst}\ntotal: {dict(total)}\nflags: {flags}")


if __name__ == "__main__":
    main()
