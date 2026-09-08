"""Generate cases, execute ASC kernels on an Ascend NPU and verify."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("test_data", ROOT / "docs" / "test_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, type=Path)
    p.add_argument("--cann-root", required=True)
    p.add_argument("--suite", choices=["smoke", "correctness", "fuzz", "stress", "all"], default="smoke")
    p.add_argument("--case-id", action="append", help="Exact ID or base ID prefix; may repeat")
    p.add_argument("--data-root", type=Path, default=ROOT / "test_data")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--report", type=Path, default=ROOT / "test_results.json")
    args = p.parse_args()
    if not args.exe.is_file():
        p.error("Executable does not exist; build first")
    selected = [c for c in data.cases(args.suite) if not args.case_id or
                any(c["id"] == item or c["id"].startswith(item + "_") for item in args.case_id)]
    if not selected:
        p.error("No matching cases")
    results = []
    for c in selected:
        directory = (args.data_root / c["id"]).resolve()
        if not directory.exists():
            data.generate(c, args.data_root.resolve())
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if any(meta[k] != v for k, v in c.items()):
            raise ValueError(f"Stale metadata: {directory}")
        (directory / "y.bin").unlink(missing_ok=True)
        command = ["bash", str(Path(__file__).with_name("launch.sh").resolve()), args.cann_root,
                   str(args.exe.resolve()), str(directory), *[str(c[k]) for k in ["B", "M", "N", "K"]],
                   c["dtype"], str(int(c["transposeX1"])), str(int(c["transposeX2"])), str(args.repeats)]
        print(f"RUN {c['id']}", flush=True)
        start = time.monotonic()
        entry = dict(id=c["id"], passed=False)
        try:
            with (directory / "execution.log").open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if result.returncode:
                raise RuntimeError(f"Executable exited {result.returncode}; see execution.log")
            data.verify(directory)
            entry["passed"] = True
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            entry["error"] = str(error)
            print(f"FAIL {c['id']}: {error}", flush=True)
        entry["wall_seconds"] = time.monotonic() - start
        results.append(entry)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Passed {sum(r['passed'] for r in results)}/{len(results)}; report={args.report}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
