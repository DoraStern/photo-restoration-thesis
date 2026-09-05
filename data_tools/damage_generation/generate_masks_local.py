"""
Local (VS Code / terminal) equivalent of the Kaggle mask-generation cell.

Generates synthetic damage masks into SEPARATE folders per damage type
(e.g. ./data/generated_masks/scratches/, ./data/generated_masks/smut/),
matching the multi-folder --masks-dir convention the training scripts use.

Idempotent per type: if a type's folder already has enough masks, that
type is skipped. If it has some but not enough, only the shortfall is
generated (existing files are never touched or overwritten -- each mask
filename includes a random ID, so repeated runs just add more).

One-time setup (only needs doing once):
    git clone --depth 1 https://github.com/daniela997/FilmDamageSimulator.git
    (then place generate_synthetic_only.py inside FilmDamageSimulator/damage_generator/)

Usage:
    python generate_masks_local.py --types scratches,smut --target-n 3000 --out-dir ./data/generated_masks
    python generate_masks_local.py --types scratches --target-n 3000   # just one type
"""

import argparse
import os
import subprocess
import sys


def find_generator_script(simulator_dir):
    """
    Different people end up with different local layouts, so check the
    plausible ones instead of assuming exactly one:
      1. <simulator_dir>/damage_generator/generate_synthetic_only.py
         -- the standard layout if simulator_dir is a full FilmDamageSimulator clone root
      2. <simulator_dir>/generate_synthetic_only.py
         -- if simulator_dir IS ALREADY the damage_generator folder itself
         (e.g. you placed this script directly inside damage_generator/
         alongside generate_synthetic_only.py, or passed --simulator-dir
         pointing straight at that folder)

    Returns (script_dir, script_path). Raises FileNotFoundError with a
    clear explanation of both locations checked if neither has the file.
    """
    candidate_1 = os.path.join(simulator_dir, "damage_generator", "generate_synthetic_only.py")
    candidate_2 = os.path.join(simulator_dir, "generate_synthetic_only.py")

    if os.path.exists(candidate_1):
        return os.path.join(simulator_dir, "damage_generator"), candidate_1
    if os.path.exists(candidate_2):
        return simulator_dir, candidate_2

    raise FileNotFoundError(
        f"Could not find generate_synthetic_only.py in either of these locations:\n"
        f"  1. {os.path.abspath(candidate_1)}\n"
        f"  2. {os.path.abspath(candidate_2)}\n"
        f"\n"
        f"--simulator-dir is currently: {os.path.abspath(simulator_dir)}\n"
        f"\n"
        f"generate_synthetic_only.py also needs its sibling files from the FilmDamageSimulator "
        f"repo in the SAME folder as it (scans.py, helpers.py, unit_converter.py), and a "
        f"synthetic/ folder containing the real damage-patch assets as a SIBLING of whatever "
        f"folder generate_synthetic_only.py sits in -- not just the single .py file on its own. "
        f"If you haven't already, clone the full repo:\n"
        f"  git clone --depth 1 https://github.com/daniela997/FilmDamageSimulator.git\n"
        f"then copy generate_synthetic_only.py into its damage_generator/ folder, and point "
        f"--simulator-dir at the FilmDamageSimulator folder itself (the one containing both "
        f"damage_generator/ and synthetic/)."
    )


def generate_type(damage_type, target_n, out_dir, simulator_dir, height, width, min_count, max_count):
    type_dir = os.path.join(out_dir, damage_type)
    os.makedirs(type_dir, exist_ok=True)

    existing = [f for f in os.listdir(type_dir) if f.startswith("mask_")]
    if len(existing) >= target_n:
        print(f"[{damage_type}] {len(existing)} masks already present at {type_dir}, skipping.")
        return

    needed = target_n - len(existing)
    print(f"[{damage_type}] {len(existing)} present, generating {needed} more...")

    generator_dir, script_path = find_generator_script(simulator_dir)

    abs_type_dir = os.path.abspath(type_dir)

    cmd = [
        sys.executable, "generate_synthetic_only.py",
        "--types", damage_type,
        "--height", str(height),
        "--width", str(width),
        "--min-count", str(min_count),
        "--max-count", str(max_count),
        "--n", str(needed),
        "--out-dir", abs_type_dir,
        "--verbose",
    ]

    subprocess.run(cmd, cwd=generator_dir, check=True)

    n_masks = len([f for f in os.listdir(type_dir) if f.startswith("mask_")])
    print(f"[{damage_type}] {n_masks} usable masks now ready at {type_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate damage masks locally, into separate per-type folders.")
    parser.add_argument("--types", type=str, default="scratches,smut",
                         help="comma-separated damage types; each gets its OWN subfolder under --out-dir")
    parser.add_argument("--target-n", type=int, default=3000, help="target number of masks per type")
    parser.add_argument("--out-dir", type=str, default="./data/generated_masks",
                         help="base folder -- each type gets a subfolder here, "
                              "e.g. ./data/generated_masks/scratches")
    parser.add_argument("--simulator-dir", type=str, default="./FilmDamageSimulator",
                         help="path to your local FilmDamageSimulator clone")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-count", type=int, default=15)
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]

    for damage_type in types:
        generate_type(
            damage_type, args.target_n, args.out_dir, args.simulator_dir,
            args.height, args.width, args.min_count, args.max_count,
        )

    print("\nSummary:")
    for damage_type in types:
        type_dir = os.path.join(args.out_dir, damage_type)
        n = len([f for f in os.listdir(type_dir) if f.startswith("mask_")]) if os.path.isdir(type_dir) else 0
        print(f"  {damage_type}: {n} masks at {type_dir}")


if __name__ == "__main__":
    main()
