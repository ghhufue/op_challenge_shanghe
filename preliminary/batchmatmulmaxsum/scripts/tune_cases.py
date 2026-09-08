"""Benchmark every implemented tiling key with the existing case runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


OP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_ROOT))

from tuning.catalog import load_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--cann-root", required=True)
    parser.add_argument(
        "--suite", choices=["smoke", "correctness", "fuzz", "stress", "all"],
        default="smoke",
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--key", action="append", type=int, help="Implemented key to test; may repeat")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=OP_ROOT / "tuning_results" / "benchmark.json",
    )
    args = parser.parse_args()

    implemented = {item.key: item for item in load_catalog() if item.implemented}
    keys = args.key or sorted(implemented)
    unknown = [key for key in keys if key not in implemented]
    if unknown:
        parser.error(f"keys are unknown or not implemented: {unknown}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = []
    for key in keys:
        key_report = args.output.with_name(f"{args.output.stem}.key_{key}.json")
        command = [
            sys.executable,
            str(OP_ROOT / "scripts" / "run_cases.py"),
            "--exe", str(args.exe),
            "--cann-root", args.cann_root,
            "--suite", args.suite,
            "--repeats", str(args.repeats),
            "--warmups", str(args.warmups),
            "--timeout", str(args.timeout),
            "--tiling-key", str(key),
            "--report", str(key_report),
        ]
        for case_id in args.case_id or []:
            command.extend(["--case-id", case_id])
        print(f"BENCH key={key} name={implemented[key].name}", flush=True)
        completed = subprocess.run(command, check=False)
        if key_report.exists():
            rows = json.loads(key_report.read_text(encoding="utf-8"))
            for row in rows:
                row["candidate_key"] = key
                row["candidate_name"] = implemented[key].name
            combined.extend(rows)
            args.output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        if completed.returncode:
            print(f"key {key} had failing cases; continuing with remaining keys", flush=True)

    passed = sum(bool(row.get("passed")) for row in combined)
    print(f"Collected {len(combined)} rows, {passed} passed; output={args.output}")
    return 0 if combined and passed == len(combined) else 1


if __name__ == "__main__":
    raise SystemExit(main())
