#!/usr/bin/env python3
"""npu_fix12_smoothquant_20260902: inspect/synthesize RKNPU2 smoothquant
stats files.

There is deliberately NO "convert calibration dump to smooth file" mode
here: GGML_RKNPU2_CALIB's dump and GGML_RKNPU2_SMOOTH's load share the
exact same file format (one line per tensor: "<name> <K> <v0> <v1> ...
<v(K-1)>", raw per-input-channel max|A_k|, see
ggml/src/ggml-rknpu2/rknpu2-smoothquant.cpp). A GGML_RKNPU2_CALIB run's
output file can be fed directly as GGML_RKNPU2_SMOOTH's input -- the C++
side (compute_and_cache_s(), called from
ggml_backend_rknpu_buffer_set_tensor()) computes s_k itself at model-load
time from that raw max|A_k| plus the max|W_k| it derives from the weight
tensor it already has in hand, per the alpha-blend formula in
rknpu2-smoothquant.h. So this tool only has two jobs, both diagnostic:

  inspect <path>          Sanity-check a stats file before using it as
                           GGML_RKNPU2_SMOOTH's input: per-tensor K, and the
                           min/max/mean of its max|A_k| values, so an
                           obviously-truncated or empty calibration run is
                           caught before a multi-hour PPL/bench pass.

  synth <path> <name> <K> Write a synthetic single-tensor stats file for
                           reproducing fix9's outlier NMSE diagnostic under
                           GGML_RKNPU2_SMOOTH without a real
                           GGML_RKNPU2_CALIB run: baseline max|A_k|=1.0 for
                           every channel, --outlier-mult (default 50.0)
                           times that on the --outlier-cols channels. See
                           npu_fix12_smoothquant_20260902.md sec "Validation
                           recipe" for how to point test-backend-ops'
                           outlier case at this file.

Usage:
    rknpu2_smooth_stats.py inspect <path>
    rknpu2_smooth_stats.py synth <path> <tensor_name> <K> \\
        [--outlier-cols 3,17,44] [--outlier-mult 50.0] [--baseline 1.0]
"""
import argparse
import sys


def read_stats(path):
    """Yields (name, K, values) per line, in the format both
    dump_calibration() (C++ writer) and load_smooth_stats_locked() (C++
    reader) use. Raises ValueError on a malformed line -- matches the C++
    loader's own truncated-line handling (stop and warn, don't half-load)."""
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"line {lineno}: expected '<name> <K> <v0> ...', got {len(parts)} field(s)")
            name = parts[0]
            try:
                k = int(parts[1])
            except ValueError:
                raise ValueError(f"line {lineno}: K field {parts[1]!r} is not an integer")
            vals_str = parts[2:]
            if len(vals_str) != k:
                raise ValueError(f"line {lineno}: tensor '{name}' declares K={k} but has {len(vals_str)} value(s)")
            yield name, k, [float(v) for v in vals_str]


def cmd_inspect(args):
    n_tensors = 0
    try:
        for name, k, vals in read_stats(args.path):
            n_tensors += 1
            lo, hi = min(vals), max(vals)
            mean = sum(vals) / len(vals) if vals else 0.0
            n_zero = sum(1 for v in vals if v == 0.0)
            flag = "  <-- ALL-ZERO (dead channel row, or never exercised by the calib corpus)" if hi == 0.0 else ""
            print(f"{name}: K={k} min={lo:.6g} max={hi:.6g} mean={mean:.6g} zero_channels={n_zero}/{k}{flag}")
    except ValueError as e:
        print(f"error: {args.path}: {e}", file=sys.stderr)
        return 1
    if n_tensors == 0:
        print(f"warning: {args.path} has no tensor stats -- GGML_RKNPU2_SMOOTH would be a pure no-op (s_k=1.0 everywhere) if pointed at this file", file=sys.stderr)
        return 1
    print(f"\n{n_tensors} tensor(s) total")
    return 0


def cmd_synth(args):
    cols = set()
    if args.outlier_cols:
        cols = {int(c) for c in args.outlier_cols.split(",") if c != ""}
    bad = [c for c in cols if not (0 <= c < args.K)]
    if bad:
        print(f"error: --outlier-cols {bad} out of range for K={args.K}", file=sys.stderr)
        return 1

    vals = [args.baseline * (args.outlier_mult if k in cols else 1.0) for k in range(args.K)]
    with open(args.path, "w") as f:
        f.write(f"{args.name} {args.K} " + " ".join(f"{v:.6g}" for v in vals) + "\n")

    print(f"wrote synthetic max|A_k| stats for '{args.name}' (K={args.K}, "
          f"{len(cols)} outlier column(s) at x{args.outlier_mult}, baseline={args.baseline}) to {args.path}")
    print("point GGML_RKNPU2_SMOOTH at this file to reproduce fix9's outlier fold in isolation "
          "(no real calibration run needed) -- see npu_fix12_smoothquant_20260902.md")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="sanity-check a stats file")
    p_inspect.add_argument("path")
    p_inspect.set_defaults(func=cmd_inspect)

    p_synth = sub.add_parser("synth", help="write a synthetic single-tensor stats file")
    p_synth.add_argument("path")
    p_synth.add_argument("name", help="weight tensor name (must match the ggml_tensor::name the outlier test case uses)")
    p_synth.add_argument("K", type=int)
    p_synth.add_argument("--outlier-cols", default="", help="comma-separated input-channel indices to inflate")
    p_synth.add_argument("--outlier-mult", type=float, default=50.0)
    p_synth.add_argument("--baseline", type=float, default=1.0)
    p_synth.set_defaults(func=cmd_synth)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
