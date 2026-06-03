#!/usr/bin/env python3
"""Convert legacy Amat pickle files to the split-format storage (json + npz + xov files)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


# Ensure local imports work when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _base_from_path(path: Path) -> Path:
    return path.with_suffix("") if path.suffix == ".pkl" else path


def _existing_split_artifacts(base: Path) -> list[Path]:
    candidates = [
        base.with_suffix(".json"),
        Path(str(base) + "_mat.npz"),
        Path(str(base) + "_mat_nosol.npz"),
    ]
    return [p for p in candidates if p.exists()]


def _resolve_out_base(input_pkl: Path, out_arg: str | None, multiple_inputs: bool) -> Path:
    if out_arg is None:
        return _base_from_path(input_pkl)

    out_path = Path(out_arg)
    if multiple_inputs:
        out_dir = out_path
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"For multiple inputs, --out must be a directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / _base_from_path(input_pkl).name

    if out_path.exists() and out_path.is_dir():
        return out_path / _base_from_path(input_pkl).name

    if out_path.suffix == ".pkl":
        return _base_from_path(out_path)

    return out_path


def convert_one(input_path: Path, out_base: Path, overwrite: bool) -> Path:
    from accumxov.Amat import Amat

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    existing = _existing_split_artifacts(out_base)
    if existing and not overwrite:
        raise FileExistsError(
            f"Split-format artifacts already exist for '{out_base}'. "
            "Use --overwrite to reconvert. Existing: "
            + ", ".join(str(p) for p in existing)
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    return Path(Amat.migrate_legacy(str(input_path), out_path=str(out_base)))


def convert_with_missing_from_nosol(
    solved_path: Path,
    nosol_path: Path,
    out_base: Path,
    overwrite: bool,
    fill_keys: set[str] | None,
) -> tuple[Path, list[str]]:
    from accumxov.Amat import Amat

    if not solved_path.exists():
        raise FileNotFoundError(f"Input file not found: {solved_path}")
    if not nosol_path.exists():
        raise FileNotFoundError(f"--fill-from-nosol file not found: {nosol_path}")

    existing = _existing_split_artifacts(out_base)
    if existing and not overwrite:
        raise FileExistsError(
            f"Split-format artifacts already exist for '{out_base}'. "
            "Use --overwrite to reconvert. Existing: "
            + ", ".join(str(p) for p in existing)
        )

    with open(solved_path, "rb") as f:
        solved_obj = pickle.load(f)
    with open(nosol_path, "rb") as f:
        nosol_obj = pickle.load(f)

    if not isinstance(solved_obj, Amat):
        raise TypeError(f"Expected Amat in {solved_path}, got {type(solved_obj)}")
    if not isinstance(nosol_obj, Amat):
        raise TypeError(f"Expected Amat in {nosol_path}, got {type(nosol_obj)}")

    merged_keys = []
    for key, nosol_value in nosol_obj.__dict__.items():
        if key == "xov":
            continue
        if fill_keys is not None and key not in fill_keys:
            continue
        solved_value = getattr(solved_obj, key, None)
        if solved_value is None and nosol_value is not None:
            setattr(solved_obj, key, nosol_value)
            merged_keys.append(key)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    solved_obj.save(str(out_base))
    return out_base, sorted(merged_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy Amat pickle file(s) to split format "
            "(base.json, base_mat.npz, base_mat_nosol.npz, xov_*.json/parquet)."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input legacy Amat pickle file(s).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help=(
            "Output base path. For one input: file base (or .pkl path). "
            "For multiple inputs: output directory. "
            "Default: same base as input path."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split-format artifacts for the target base path.",
    )
    parser.add_argument(
        "--fill-from-nosol",
        default=None,
        help=(
            "Optional legacy *_nosol.pkl to fill only fields missing in the main input "
            "before writing split artifacts (single input only)."
        ),
    )
    parser.add_argument(
        "--fill-keys",
        default="spA,b,weights",
        help=(
            "Comma-separated attribute names allowed to be copied from --fill-from-nosol. "
            "Default: spA,b,weights. Use 'b,weights' to mimic leaner new-run outputs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(p).expanduser() for p in args.inputs]
    nosol_path = Path(args.fill_from_nosol).expanduser() if args.fill_from_nosol else None
    fill_keys = {
        x.strip() for x in str(args.fill_keys).split(",") if x.strip()
    } if nosol_path else None
    had_error = False

    if nosol_path and len(input_paths) != 1:
        print(
            "ERROR: --fill-from-nosol supports exactly one input file.",
            file=sys.stderr,
        )
        return 1

    for input_path in input_paths:
        try:
            out_base = _resolve_out_base(input_path, args.out, len(input_paths) > 1)
            if nosol_path:
                migrated_base, merged_keys = convert_with_missing_from_nosol(
                    solved_path=input_path,
                    nosol_path=nosol_path,
                    out_base=out_base,
                    overwrite=args.overwrite,
                    fill_keys=fill_keys,
                )
                print(
                    f"OK: {input_path} + {nosol_path} -> {migrated_base}.json "
                    f"(filled missing fields: {', '.join(merged_keys) if merged_keys else 'none'})"
                )
            else:
                migrated_base = convert_one(input_path, out_base, overwrite=args.overwrite)
                print(f"OK: {input_path} -> {migrated_base}.json")
        except Exception as exc:
            had_error = True
            print(f"ERROR: {input_path}: {exc}", file=sys.stderr)

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
