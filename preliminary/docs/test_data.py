"""Reproducible local test specification; not the official competition cases.

Examples (from repository root):
  python preliminary/docs/test_data.py list
  python preliminary/docs/test_data.py self-check
  python preliminary/docs/test_data.py generate --suite smoke --out preliminary/test_data
  python preliminary/docs/test_data.py verify --case-dir <generated-case-directory>
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16


# Each record: name, (B, M, N, K), input pattern.
SUITES = {
    "smoke": [
        ("s00_dot", (1, 1, 1, 32), "uniform"),
        ("s01_known", (1, 2, 3, 32), "known"),
        ("s02_negative", (1, 2, 3, 32), "negative"),
        ("s03_order", (1, 2, 2, 32), "order"),
        ("s04_batch", (3, 3, 5, 40), "batch"),
    ],
    "correctness": [
        ("c00_zero", (2, 17, 33, 40), "zero"),
        ("c01_one", (2, 17, 33, 40), "one"),
        ("c02_negative_tail", (2, 17, 33, 40), "negative"),
        ("c03_last_n", (2, 17, 257, 40), "last_n"),
        ("c04_tie", (2, 17, 65, 40), "tie"),
        ("c05_near_tie", (2, 17, 65, 40), "near_tie"),
        ("c06_k_cancel", (1, 2, 2, 8192), "k_cancel"),
        ("c07_sum_cancel", (2, 33, 17, 40), "sum_cancel"),
        ("c08_range", (2, 17, 33, 40), "range"),
        ("c09_m1", (1, 1, 1025, 72), "uniform"),
        ("c10_n1", (1, 1025, 1, 72), "uniform"),
        ("c11_b64", (64, 3, 5, 40), "batch"),
        ("c12_align16", (2, 16, 16, 32), "uniform"),
        ("c13_tail15", (2, 15, 15, 40), "uniform"),
        ("c14_tail17", (2, 17, 17, 72), "uniform"),
        ("c15_tail31", (2, 31, 33, 120), "uniform"),
        ("c16_tail63", (2, 63, 65, 128), "uniform"),
        ("c17_tail127", (2, 127, 129, 264), "uniform"),
        ("c18_tail255", (1, 255, 257, 520), "uniform"),
        ("c19_rect_m", (1, 513, 33, 256), "uniform"),
        ("c20_rect_n", (1, 33, 513, 256), "uniform"),
        ("c21_square", (4, 128, 128, 128), "normalized"),
        ("c22_large_k", (1, 17, 19, 8192), "uniform"),
        ("c23_large_k_tail", (1, 17, 19, 8184), "uniform"),
        ("c24_negative_multitile", (2, 17, 257, 40), "negative"),
        ("c25_bmn_first_group", (2, 17, 513, 64), "first_n"),
        ("c26_bmn_group_winners", (2, 17, 1025, 64), "n_group_winners"),
    ],
    "stress": [
        ("t00_square", (1, 2048, 2048, 128), "uniform"),
        ("t01_max_mn", (1, 8192, 8192, 32), "uniform"),
        ("t02_max_mk", (1, 8192, 1, 8192), "uniform"),
        ("t03_max_nk", (1, 1, 8192, 8192), "uniform"),
        ("t04_batch_cap", (64, 128, 128, 8192), "uniform"),
        ("t05_all_max", (1, 8192, 8192, 8192), "uniform"),
        ("t06_max_mn_tail", (1, 8191, 8191, 40), "uniform"),
        ("t07_similarity_cap", (64, 8192, 8192, 128), "uniform"),
    ],
}


def records(suite):
    if suite == "fuzz":
        rng = np.random.default_rng(20260907)
        edges = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129]
        return [(f"f{i:02d}", (int(rng.choice([1, 2, 3, 7])),
                 int(rng.choice(edges)), int(rng.choice(edges)),
                 int(rng.choice([32, 40, 56, 64, 72, 120, 128, 264]))), "uniform")
                for i in range(64)]
    return SUITES[suite]


def cases(suite):
    for group in (list(SUITES) + ["fuzz"] if suite == "all" else [suite]):
        for name, shape, pattern in records(group):
            b, m, n, k = shape
            assert 1 <= b <= 64 and 1 <= m <= 8192 and 1 <= n <= 8192
            assert 32 <= k <= 8192 and k % 8 == 0
            assert b * m * k <= 2**26 and b * n * k <= 2**26
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            for dtype in ["float16", "bfloat16"]:
                for tx1, tx2 in [(False, False), (False, True), (True, False), (True, True)]:
                    yield dict(id=f"{name}_{dtype}_{int(tx1)}{int(tx2)}", suite=group,
                               B=b, M=m, N=n, K=k, pattern=pattern, seed=seed,
                               dtype=dtype, transposeX1=tx1, transposeX2=tx2)


def logical_inputs(case):
    b, m, n, k = (case[key] for key in ["B", "M", "N", "K"])
    shape_a, shape_d = (b, m, k), (b, n, k)
    rng = np.random.default_rng(case["seed"])
    pattern = case["pattern"]
    expected = None
    if pattern in ["uniform", "normalized", "range"]:
        a = rng.uniform(-1, 1, shape_a).astype(np.float32)
        d = rng.uniform(-1, 1, shape_d).astype(np.float32)
        if pattern == "normalized":
            a /= np.linalg.norm(a, axis=-1, keepdims=True)
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
        elif pattern == "range":
            scales = np.resize(np.array([2**-8, 1, 2**8], np.float32), k)
            a *= scales
            d /= scales
    else:
        a, d = np.zeros(shape_a, np.float32), np.zeros(shape_d, np.float32)
        if pattern == "zero":
            expected = np.zeros(b)
        elif pattern == "one":
            a.fill(1)
            d.fill(1)
            expected = np.full(b, m * k)
        elif pattern == "known":
            a[:, 0, 0], a[:, 1, 1] = 1, 1
            d[:, :, :2] = [[1, 0], [0, 1], [-1, 0]]
            expected = np.full(b, 2)
        elif pattern == "negative":
            a[:, :, 0] = 1
            d[:, :, 0] = -(1 + np.arange(n) % 7)
            expected = np.full(b, -m)
        elif pattern == "order":
            a[:, 0, 0], a[:, 1, 1] = 1, 1
            d[:, :, :2] = [[10, 0], [0, 10]]
            expected = np.full(b, 20)
        elif pattern == "batch":
            a[:, :, 0] = np.arange(1, b + 1)[:, None]
            d[:, :, 0] = np.arange(1, b + 1)[:, None] / 8
            expected = m * np.arange(1, b + 1, dtype=np.float64)**2 / 8
        elif pattern in ["first_n", "last_n", "tie", "near_tie"]:
            a[:, :, 0] = 1
            if pattern == "first_n":
                d[:, :, 0], d[:, 0, 0] = -1, 2
                expected = np.full(b, 2 * m)
            elif pattern == "last_n":
                d[:, :, 0], d[:, -1, 0] = -1, 2
                expected = np.full(b, 2 * m)
            else:
                d[:, :, 0] = 1
                if pattern == "near_tie":
                    d[:, -1, 0] = 1 + 2**-7  # Distinct in both FP16 and BF16.
                expected = np.full(b, m * (1 + (2**-7 if pattern == "near_tie" else 0)))
        elif pattern == "n_group_winners":
            d.fill(-1)
            for row in range(m):
                depth = row % k
                anchor = min((row % 4) * 256, n - 1)
                a[:, row, depth] = 1
                d[:, anchor, depth] = 2
            expected = np.full(b, 2 * m)
        elif pattern == "k_cancel":
            a.fill(1)
            d[:, 0, :k // 2], d[:, 0, k // 2:] = 1, -1
            d[:, 1, 0] = 1
            expected = np.full(b, m)
        elif pattern == "sum_cancel":
            a[:, :, 0] = np.where(np.arange(m) % 2 == 0, 1, -1)
            a[:, -1, 0] = 2**-10
            d[:, :, 0] = 1
            expected = np.full(b, 2**-10)
        else:
            raise ValueError(pattern)
    dtype = np.float16 if case["dtype"] == "float16" else bfloat16
    # Quantize before golden. Four layouts reuse exactly the same logical values.
    return a.astype(dtype), d.astype(dtype), expected


def golden(a, d, tile_m=64, tile_n=256):
    """Full-K FP64 dot products; block only M/N to limit similarity memory."""
    b, m, _ = a.shape
    n = d.shape[1]
    result = np.empty(b, np.float32)
    for batch in range(b):
        rows = np.empty(m, np.float64)
        for mi in range(0, m, tile_m):
            aa = a[batch, mi:mi + tile_m].astype(np.float64)
            best = np.full(len(aa), -np.inf)
            for ni in range(0, n, tile_n):
                dd = d[batch, ni:ni + tile_n].astype(np.float64)
                best = np.maximum(best, (aa @ dd.T).max(axis=1))
            rows[mi:mi + len(aa)] = best
        result[batch] = rows.sum(dtype=np.float64)
    assert np.isfinite(result).all()
    return result


def generate(case, root):
    directory = root / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    a, d, expected = logical_inputs(case)
    y = golden(a, d)
    if expected is not None:
        np.testing.assert_array_equal(y, expected.astype(np.float32))
    x1 = np.ascontiguousarray(a.swapaxes(1, 2) if case["transposeX1"] else a)
    x2 = np.ascontiguousarray(d if case["transposeX2"] else d.swapaxes(1, 2))
    meta = dict(case, x1_shape=list(x1.shape), x2_shape=list(x2.shape),
                y_shape=[case["B"]], y_dtype="float32", byte_order=sys.byteorder,
                numpy_version=np.__version__, rtol=1e-4, atol=1e-4, files={})
    for name, array in [("x1.bin", x1), ("x2.bin", x2), ("golden_y.bin", y)]:
        path = directory / name
        array.tofile(path)
        with path.open("rb") as binary:
            digest = hashlib.file_digest(binary, "sha256").hexdigest()
        meta["files"][name] = dict(bytes=path.stat().st_size,
                                  sha256=digest)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def self_check():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "batchmatmulmaxsum" / "scripts"))
    from BatchMatmulMaxSum import impl
    all_cases = list(cases("all"))
    assert len({c["id"] for c in all_cases}) == len(all_cases)
    count = 0
    # Every semantic pattern, both dtypes, four layouts; tiny tiles force multiple rounds.
    selected = [c for c in all_cases if c["suite"] == "smoke" or
                (c["suite"] == "correctness" and c["pattern"] != "uniform")]
    for c in selected:
        a, d, expected = logical_inputs(c)
        y = golden(a, d, 7, 11)
        x1 = np.ascontiguousarray(a.swapaxes(1, 2) if c["transposeX1"] else a)
        x2 = np.ascontiguousarray(d if c["transposeX2"] else d.swapaxes(1, 2))
        ref = impl(x1, x2, c["transposeX1"], c["transposeX2"])
        np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)
        if expected is not None:
            np.testing.assert_array_equal(y, expected.astype(np.float32))
        count += 1
    print(f"Validated {len(all_cases)} case definitions; {count} CPU golden/layout checks passed.")


def verify(directory):
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    expected_bytes = 4 * meta["B"]
    for name in ["y.bin", "golden_y.bin"]:
        if (directory / name).stat().st_size != expected_bytes:
            raise ValueError(f"{name}: expected exactly {expected_bytes} bytes")
    actual = np.fromfile(directory / "y.bin", dtype=np.float32)
    expected = np.fromfile(directory / "golden_y.bin", dtype=np.float32)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite output or golden")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bound = meta["atol"] + meta["rtol"] * np.abs(expected.astype(np.float64))
    if not np.all(error <= bound):
        raise ValueError(f"Numerical mismatch: max_abs_error={error.max()}")
    print(f"PASS {meta['id']}: max_abs_error={error.max()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "generate", "self-check", "verify"])
    parser.add_argument("--suite", choices=list(SUITES) + ["fuzz", "all"], default="smoke")
    parser.add_argument("--out", type=Path, default=Path("preliminary/test_data"))
    parser.add_argument("--case-id")
    parser.add_argument("--case-dir", type=Path)
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
        return
    if args.action == "verify":
        if args.case_dir is None:
            parser.error("verify requires --case-dir")
        verify(args.case_dir)
        return
    selected = list(cases(args.suite))
    if args.case_id:
        selected = [c for c in selected if c["id"] == args.case_id]
        if not selected:
            parser.error("case ID not found in selected suite")
    if args.action == "list":
        print(json.dumps(selected, indent=2))
        return
    if sys.byteorder != "little":
        parser.error("Binary generation currently supports little-endian hosts only")
    for c in selected:
        print(f"Generating {c['id']}", flush=True)
        generate(c, args.out)
    print(f"Generated {len(selected)} cases under {args.out}")


if __name__ == "__main__":
    main()
