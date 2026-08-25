"""
Standalone CLI: composite a generated damage mask onto a clean target
image, producing a damaged/clean pair for quick visual checks.

The actual blend math lives in common/blend.py, shared with
DegradedPairDataset and VAE2PairDataset -- this file is just a thin CLI
wrapper around it now, not a separate implementation.

Usage:
    python composite_damage.py --clean path/to/clean.png --mask path/to/mask.png --out path/to/damaged.png
    python composite_damage.py --clean path/to/clean.png --mask path/to/mask.png --out path/to/damaged.png --blend multiply
"""

import argparse
import cv2 as cv

from common.blend import composite_numpy as composite


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Composite a damage mask onto a clean image.')
    parser.add_argument('--clean', required=True, help='path to the clean input image')
    parser.add_argument('--mask', required=True, help='path to the generated grayscale damage mask')
    parser.add_argument('--out', required=True, help='path to write the damaged output image')
    parser.add_argument('--blend', choices=['screen', 'multiply'], default='screen',
                         help="'screen' (default) produces light/white damage marks; "
                              "'multiply' produces dark damage marks")
    args = parser.parse_args()

    clean_img = cv.imread(args.clean, cv.IMREAD_UNCHANGED)
    mask_img = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)

    damaged = composite(clean_img, mask_img, blend=args.blend)
    cv.imwrite(args.out, damaged)
    print(f"Wrote damaged image to {args.out} (blend={args.blend})")
