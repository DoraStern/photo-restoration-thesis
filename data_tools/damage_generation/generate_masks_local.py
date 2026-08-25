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


def generate_type(damage_type, target_n, out_dir, simulator_dir, height, width, min_count, max_count):
    type_dir = os.path.join(out_dir, damage_type)
    os.makedirs(type_dir, exist_ok=True)

    existing = [f for f in os.listdir(type_dir) if f.startswith("mask_")]
    if len(existing) >= target_n:
        print(f"[{damage_type}] {len(existing)} masks already present at {type_dir}, skipping.")
        return

    needed = target_n - len(existing)
    print(f"[{damage_type}] {len(existing)} present, generating {needed} more...")

    generator_dir = os.path.join(simulator_dir, "damage_generator")
    script_path = os.path.join(generator_dir, "generate_synthetic_only.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Could not find {script_path}. Clone FilmDamageSimulator and copy "
            f"generate_synthetic_only.py into its damage_generator/ folder first -- see this "
            f"script's docstring for the one-time setup command."
        )

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