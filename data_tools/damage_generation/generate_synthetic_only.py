"""
Generate damage overlay masks using ONLY the pre-classified synthetic damage
patches in /synthetic/<type>/ (e.g. scratches, smut), without ever touching
the real scanned film frames in /scans/.

This bypasses damage_generator.py's default behaviour, which always loads
/scans/ and mixes real scanned artifact crops into the sampling pool even
when --synthetic is passed. Here, only the folder(s) you name are loaded,
and artifact count/size statistics are fit on those patches' own area
distribution instead of the real-scan-derived Gamma distributions.

Usage:
    python generate_synthetic_only.py --types scratches,smut --height 1024 --width 1024
    python generate_synthetic_only.py --types scratches --procedural-scratches
"""

import os
import argparse
import uuid
import random
import numpy as np
import pandas as pd
import cv2 as cv
import scipy.stats as stats
import skimage.transform as skimage_tf

from scans import load_images
from generate_masks import generate_perlin_noise_2d, increase_contrast, random_perlin_with_numpy, line_scratch


def sample_size_from_own_distribution(df, num_artifact):
    """Fit a Gamma distribution to this dataframe's OWN artifact areas
    (instead of a real-scan-derived one) and sample target sizes from it."""
    areas = df['Contour Area']
    gamma_param = stats.gamma.fit(areas, floc=0)
    shape, _, scale = gamma_param
    return np.random.gamma(shape, scale, num_artifact)


def sample_closest_in_area(df, target_areas):
    df = df.sample(frac=1).reset_index(drop=True)
    areas = df['Contour Area']
    indexes = []
    for target in target_areas:
        candidates = df.iloc[(areas - target).abs().argsort()[:15]].index.tolist()
        index = random.choice(candidates)
        indexes.append(index)
        areas = areas.drop(areas.index[[index]])
    picked = df.iloc[indexes].copy()
    picked['Target size'] = target_areas
    return picked


def build_mask(target_size, per_type_dfs, per_type_counts, rescale=True, verbose=False):
    rescale_factor = (target_size[0] / 2560 if target_size[0] <= target_size[1]
                       else target_size[1] / 2560) if rescale else 1.

    selected_frames = []
    for artifact_type, df in per_type_dfs.items():
        lo, hi = per_type_counts[artifact_type]
        num = int(np.random.randint(lo, hi + 1))
        if num == 0 or len(df) == 0:
            continue
        target_areas = sample_size_from_own_distribution(df, num)
        picked = sample_closest_in_area(df, target_areas)
        selected_frames.append(picked)
        if verbose:
            print(f"Selected {num} '{artifact_type}' artifacts")

    if not selected_frames:
        raise ValueError("No artifacts selected - check your --types and --min-count/--max-count")

    selected_artifacts_df = pd.concat(selected_frames, ignore_index=True)
    artifacts_num = len(selected_artifacts_df)

    mask_final = np.zeros(target_size).astype(np.uint8)
    perlin_noise = generate_perlin_noise_2d(target_size, (2, 2))
    normalised_noise = (perlin_noise - np.min(perlin_noise)) / np.ptp(perlin_noise)
    xs, ys = random_perlin_with_numpy(artifacts_num, normalised_noise)
    random_angles = np.random.randint(0, 360, size=artifacts_num)

    i = 0
    for _, artifact_row in selected_artifacts_df.iterrows():
        try:
            artifact = artifact_row['Artifact'].astype(np.uint8)
            random_scale = artifact_row['Target size'] / artifact_row['Contour Area']
            random_angle = random_angles[i]
            new_rescale_factor = rescale_factor * np.sqrt(random_scale)
            artifact = skimage_tf.rescale(artifact, round(new_rescale_factor, 2), anti_aliasing=True, preserve_range=True)
            artifact = skimage_tf.rotate(artifact, angle=random_angle, resize=True, preserve_range=True)
            artifact_w, artifact_h = artifact.shape[:2]

            x1 = xs[i] - artifact_w // 2
            x2 = x1 + artifact_w
            if x1 < 0:
                artifact = artifact[-x1:, :]; x1 = 0
            if x2 > target_size[0]:
                artifact = artifact[:-(x2 - target_size[0]), :]; x2 = target_size[0]

            y1 = ys[i] - artifact_h // 2
            y2 = y1 + artifact_h
            if y1 < 0:
                artifact = artifact[:, -y1:]; y1 = 0
            if y2 > target_size[1]:
                artifact = artifact[:, :-(y2 - target_size[1])]; y2 = target_size[1]

            mask_final[x1:x2, y1:y2] = np.where(
                artifact > mask_final[x1:x2, y1:y2], artifact, mask_final[x1:x2, y1:y2]
            )
            i += 1
        except Exception:
            i += 1
            continue

    mask_final = np.invert(mask_final.astype(np.uint8))
    binarised = ((mask_final > 240) * 255).astype(np.uint8)
    return mask_final.astype(np.uint8), binarised


def add_procedural_scratches(mask, height, width, verbose=False):
    """Blend in fully procedural (Perlin-noise-based) scratch lines.
    These require NO source images at all -- real or synthetic -- so they
    are always 'safe' to include without pulling in any scan data."""
    num_extra_scratch = int(np.random.gamma(6, 2, 1)[0])
    for _ in range(num_extra_scratch):
        length = np.random.randint(10, high=max(height, width), dtype=int)
        try:
            scratch = line_scratch(np.array(length))
            sw, sh = scratch.shape[:2]
            if sw >= width or sh >= height:
                continue
            x1 = np.random.randint(0, width - sw)
            y1 = np.random.randint(0, height - sh)
            region = mask[x1:x1 + sw, y1:y1 + sh]
            mask[x1:x1 + sw, y1:y1 + sh] = np.minimum(region, np.invert(scratch.astype(np.uint8)))
        except Exception:
            continue
    if verbose:
        print(f"Added {num_extra_scratch} procedural scratch lines")
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate damage masks from ONLY classified synthetic patches (no scanned frames).'
    )
    parser.add_argument('--types', type=str, default='scratches,smut',
                         help='comma-separated subfolder names under /synthetic/, '
                              'e.g. scratches,smut,dirt,dots,hair,hair-short,lint,sprinkles,spots,stain')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--min-count', type=int, default=3, help='min number of artifacts per type')
    parser.add_argument('--max-count', type=int, default=15, help='max number of artifacts per type')
    parser.add_argument('--procedural-scratches', action='store_true',
                         help='also blend in fully procedural line scratches (no source image needed)')
    parser.add_argument('--n', type=int, default=1, help='how many masks to generate')
    parser.add_argument('--out-dir', type=str, default=None,
                         help='where to write masks. Defaults to <repo_root>/generated/. Set this explicitly '
                              'to generate straight into a per-type folder, e.g. --out-dir ../../data/masks/scratches '
                              'when running with --types scratches only, so different damage types land in '
                              'physically separate folders instead of one mixed pool.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    abs_path = os.path.abspath(os.path.dirname(__file__))
    synthetic_path = os.path.dirname(os.path.normpath(abs_path)) + '/synthetic/'
    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.normpath(abs_path)) + '/generated/'
    if not out_dir.endswith('/'):
        out_dir += '/'
    os.makedirs(out_dir, exist_ok=True)

    types = [t.strip() for t in args.types.split(',') if t.strip()]

    per_type_dfs = {}
    for t in types:
        df = load_images(synthetic_path, t, verbose=args.verbose)
        df['Contour Area'] = df['Non-zero pixel area']
        per_type_dfs[t] = df
        print(f"Loaded {len(df)} '{t}' artifact patches from /synthetic/{t}/")

    per_type_counts = {t: (args.min_count, args.max_count) for t in types}

    for n in range(args.n):
        mask, binary_mask = build_mask(
            (args.height, args.width), per_type_dfs, per_type_counts, verbose=args.verbose
        )

        if args.procedural_scratches:
            mask = add_procedural_scratches(mask, args.height, args.width, verbose=args.verbose)
            binary_mask = ((mask > 240) * 255).astype(np.uint8)

        uid = str(uuid.uuid4())[:8]
        tag = "_".join(types)
        cv.imwrite(out_dir + f'mask_{tag}_{uid}.png', mask)
        cv.imwrite(out_dir + f'binarised_mask_{tag}_{uid}.png', binary_mask)
        print(f"[{n+1}/{args.n}] Saved mask_{tag}_{uid}.png")

    print(f"Done. Masks written to {out_dir}")