# Digital Photograph Restoration — Thesis Comparison

Three restoration methods, compared on the same shared data: a GAN + VAE
latent-translation method, DiffBIR-style diffusion restoration (both
stages), and a transformer regression method (Swin/SwinIR-style).

**This is a rebuilt, consolidated version of the project.** All files
were re-verified end to end together in one pass (not just individually)
before packaging, to fix drift that had crept in between earlier
individually-delivered files and an earlier repo bundle.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── common/                        # Shared, method-agnostic code
│   ├── blend.py                   #   damage compositing (screen/multiply) --
│   │                               #   single source of truth
│   ├── degraded_pair_dataset.py   #   used by Method 2 AND Method 3 (identical)
│   └── real_photo_dataset.py      #   used by Method 1 (domain A)
│
├── data_tools/                    # One-time data preparation scripts
│   ├── scrapers/
│   │   ├── loc_scraper.py                # LOC, supports weighted multi-category scraping
│   │   ├── dpla_scraper.py               # Digital Public Library of America
│   │   ├── download_kaggle_dataset.py    # pull your own uploaded Kaggle dataset
│   │   └── download_gdrive_dataset.py    # pull from a public Google Drive link
│   └── damage_generation/
│       ├── generate_synthetic_only.py    # FilmDamageSimulator mask generator (--out-dir supported)
│       ├── generate_masks_local.py       # local/VS Code wrapper, per-type folders, idempotent
│       └── composite_damage.py           # standalone compositing CLI tool
│
├── method1_gan_vae/                # GAN + VAE latent translation
│   ├── models/
│   │   ├── vae_blocks.py           #   NOTE: renamed from blocks.py -- Method 2
│   │   │                           #   has its own different blocks.py; distinct
│   │   │                           #   names avoid a real naming collision
│   │   ├── vae.py                  #   DomainVAE (shared class for VAE1 and VAE2)
│   │   └── translation_net.py      #   LatentTranslationNet + LatentDiscriminator
│   ├── data/
│   │   └── vae2_pair_dataset.py    #   clean+degraded+mask triples (VAE2-specific)
│   ├── train_vae_domain_a.py       #   Stage 1: VAE1 on real old photos
│   ├── train_vae_domain_b.py       #   Stage 2: VAE2 on clean + synthetic damage
│   ├── train_translation_net.py    #   Stage 3: the actual repair mechanism
│   └── evaluate.py                 #   TWO modes: --mode synthetic (quantitative,
│                                    #   PSNR/SSIM against known ground truth) and
│                                    #   --mode real (qualitative only -- no ground
│                                    #   truth exists for genuinely old photos)
│
├── method2_diffbir/                 # DiffBIR-style, both stages
│   ├── models/
│   │   ├── unet_blocks.py           #   NOTE: renamed from blocks.py, see above
│   │   ├── restoration_net.py       #   RestorationUNet (Stage 1 regression net)
│   │   └── diffusion_stage2.py      #   frozen SD components + ControlNet-style adapter
│   ├── train_stage1_restoration.py
│   ├── train_stage2_diffusion.py    #   requires internet access to Hugging Face Hub
│   │                                #   (run on Kaggle, not a network-sandboxed env)
│   └── evaluate.py                  #   Stage 1 only; Stage 2 has no evaluate.py yet
│
├── method3_transformer/             # Transformer regression (Swin/SwinIR-style)
│   ├── models/
│   │   └── transformer_restoration.py   # supports --use-checkpoint (gradient
│   │                                     # checkpointing, recommended at 256px+)
│   ├── train_transformer_regression.py
│   └── evaluate.py
│
└── data/                            # NOT committed to git -- see .gitignore.
    ├── real_old_photos/             #   from data_tools/scrapers/*.py
    │   ├── images/
    │   └── manifest.csv
    ├── voc2012/                     #   clean photos (VOCdevkit/VOC2012/JPEGImages)
    └── generated_masks/             #   from generate_synthetic_only.py / generate_masks_local.py
        ├── scratches/                #   keep damage types in SEPARATE folders
        └── smut/
```

## Why this structure

**One `data/` folder, shared by all three methods.** Run the scraper and
mask generator once into `data/`, and point every method's
`--clean-dir`/`--masks-dir`/`--real-photo-dir` at those same shared
folders.

**`common/blend.py` is the single source of truth for damage compositing.**
Earlier in this project the screen/multiply blend logic was duplicated in
three places (the standalone CLI tool and two dataset classes), which
meant a fix had to be applied three times. It's now one function.

**`vae_blocks.py` / `unet_blocks.py` instead of two files both named
`blocks.py`.** Method 1's VAE architecture and Method 2's U-Net
architecture each had their own `blocks.py` with completely different
contents. Distinct names remove the risk of one silently shadowing the
other.

**Masks are generated into per-damage-type folders** (`scratches/`,
`smut/`, etc.) rather than one mixed pool. Every training script's
`--masks-dir` accepts one OR MORE folders, so you can combine types or
isolate a single one per run:

```bash
--masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut
```

## Setup

```bash
pip install -r requirements.txt
```

## One-time shared data preparation

```bash
# Real old photos (Method 1 only):
pip install gdown
python download_gdrive_dataset.py --file-url "https://drive.google.com/file/d/1knqFN2K026XfFg53uKQjJlcw0yEXcpqm/view?usp=sharing" \
    --out-dir ./data/real_old_photos


# Clean photos (all three methods)
python -c "import torchvision.datasets as d; d.VOCDetection(root='./data/voc2012', year='2012', image_set='train', download=True)"

# Damage masks, into SEPARATE per-type folders (all three methods)
git clone --depth 1 https://github.com/daniela997/FilmDamageSimulator.git
cp data_tools/damage_generation/generate_synthetic_only.py FilmDamageSimulator/damage_generator/

python data_tools/damage_generation/generate_masks_local.py \
    --types scratches --target-n 3000 \
    --out-dir ./data/generated_masks --simulator-dir ./FilmDamageSimulator
    python data_tools/damage_generation/generate_masks_local.py \
    --types smut --target-n 3000 \
    --out-dir ./data/generated_masks --simulator-dir ./FilmDamageSimulator
    python data_tools/damage_generation/generate_masks_local.py \
    --types spots --target-n 3000 \
    --out-dir ./data/generated_masks --simulator-dir ./FilmDamageSimulator
    python data_tools/damage_generation/generate_masks_local.py \
    --types dirt --target-n 3000 \
    --out-dir ./data/generated_masks --simulator-dir ./FilmDamageSimulator
```

## Held-out validation and test sets

Before training, split off validation and test slices that are **never**
used in training for any method. Point each training script's
`--val-clean-dir`/`--val-masks-dir`/`--val-data-root` at the validation
split; keep the test split completely untouched until your final
cross-method comparison.

## Running each method

All commands run from the **repo root**, using `python -m` so the shared
`common/` package resolves correctly.

Each method below is self-contained: a **baseline check** first (a short
run at your real target resolution/batch size, confirming the whole
pipeline works before committing GPU-hours to a long run), then the **full
training run**, then **evaluation**. Baseline checks write to
`./runs/baseline_check/...`, kept separate from the real run's `./runs/...`
so nothing gets overwritten.

All training scripts share the same conventions: `--steps-per-epoch` to
shorten an epoch for quick tests, `--amp` for mixed precision,
`--resume <checkpoint>` to continue, `--log-every` for timing diagnostics,
`--val-*` args for validation-based `best.pt` checkpoint selection, and a
`train_log.txt` written inside each run's `--out-dir`.

---

### Method 1 — GAN + VAE (three stages)

**Baseline check** — all three stages at real config, 8 epochs each:

```bash
python -m method1_gan_vae.train_vae_domain_a \
    --data-root ./data/real_old_photos \
    --epochs 8 --batch-size 16 --image-size 256 \
    --amp --sample-every 50 --save-every 4 \
    --out-dir ./runs/baseline_check/vae_domain_a

python -m method1_gan_vae.train_vae_domain_b \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 8 --batch-size 16 --image-size 256 \
    --amp --sample-every 50 --save-every 4 \
    --out-dir ./runs/baseline_check/vae_domain_b

python -m method1_gan_vae.train_translation_net \
    --vae1-checkpoint ./runs/baseline_check/vae_domain_a/checkpoints/<latest>.pt \
    --vae2-checkpoint ./runs/baseline_check/vae_domain_b/checkpoints/<latest>.pt \
    --real-photo-dir ./data/real_old_photos \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 8 --batch-size 16 --image-size 256 \
    --sample-every 50 --save-every 4 \
    --out-dir ./runs/baseline_check/translation_net

python -m method1_gan_vae.evaluate --mode synthetic \
    --vae1-checkpoint ./runs/baseline_check/vae_domain_a/checkpoints/<latest>.pt \
    --vae2-checkpoint ./runs/baseline_check/vae_domain_b/checkpoints/<latest>.pt \
    --translation-checkpoint ./runs/baseline_check/translation_net/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 20 --out-dir ./eval_results_baseline_m1
```

**Full training run**, then evaluation, once the baseline check looks right:

```bash
python -m method1_gan_vae.train_vae_domain_a \
    --data-root ./data/real_old_photos --epochs 50 --batch-size 16 --image-size 256 \
    --amp --out-dir ./runs/vae_domain_a

python -m method1_gan_vae.train_vae_domain_b \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 50 --batch-size 16 --image-size 256 --amp --out-dir ./runs/vae_domain_b

python -m method1_gan_vae.train_translation_net \
    --vae1-checkpoint ./runs/vae_domain_a/checkpoints/<latest>.pt \
    --vae2-checkpoint ./runs/vae_domain_b/checkpoints/<latest>.pt \
    --real-photo-dir ./data/real_old_photos \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 50 --batch-size 16 --image-size 256 --out-dir ./runs/translation_net

python -m method1_gan_vae.evaluate --mode synthetic \
    --vae1-checkpoint ./runs/vae_domain_a/checkpoints/<latest>.pt \
    --vae2-checkpoint ./runs/vae_domain_b/checkpoints/<latest>.pt \
    --translation-checkpoint ./runs/translation_net/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 100 --out-dir ./eval_results_m1
```

---

### Method 2 — DiffBIR (both stages)

**Baseline check** — Stage 1 at real config, then a smaller-scale Stage 2
check (Stage 2 is far heavier per step, so its baseline check uses fewer
epochs and a smaller batch size just to confirm the mechanics work):

```bash
python -m method2_diffbir.train_stage1_restoration \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 8 --batch-size 16 --image-size 256 \
    --amp --sample-every 50 --save-every 4 \
    --out-dir ./runs/baseline_check/stage1_restoration

python -m method2_diffbir.evaluate \
    --checkpoint ./runs/baseline_check/stage1_restoration/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 20 --out-dir ./eval_results_baseline_m2

python -m method2_diffbir.train_stage2_diffusion \
    --stage1-checkpoint ./runs/baseline_check/stage1_restoration/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 2 --batch-size 1 --image-size 256 \
    --gradient-checkpointing --amp --sample-every 20 --save-every 1 \
    --out-dir ./runs/baseline_check/stage2_diffusion

python -m method2_diffbir.evaluate_stage2 \
    --stage1-checkpoint ./runs/baseline_check/stage1_restoration/checkpoints/<latest>.pt \
    --controlnet-checkpoint ./runs/baseline_check/stage2_diffusion/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 10 --num-inference-steps 20 --out-dir ./eval_results_baseline_m2_stage2
```

**Full training run**, then evaluation, once the baseline check looks right:

```bash
python -m method2_diffbir.train_stage1_restoration \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 50 --batch-size 16 --image-size 256 --amp --out-dir ./runs/stage1_restoration

python -m method2_diffbir.evaluate \
    --checkpoint ./runs/stage1_restoration/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 100 --out-dir ./eval_results_m2

python -m method2_diffbir.train_stage2_diffusion \
    --stage1-checkpoint ./runs/stage1_restoration/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 20 --batch-size 1 --image-size 256 \
    --gradient-checkpointing --amp --out-dir ./runs/stage2_diffusion

python -m method2_diffbir.evaluate_stage2 \
    --stage1-checkpoint ./runs/stage1_restoration/checkpoints/<latest>.pt \
    --controlnet-checkpoint ./runs/stage2_diffusion/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 100 --num-inference-steps 20 --out-dir ./eval_results_m2_full
```

---

### Method 3 — Transformer Regression

**Baseline check**:

```bash
python -m method3_transformer.train_transformer_regression \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 8 --batch-size 4 --image-size 256 \
    --embed-dim 60 --depths "4,4,4,4" --num-heads 6 --window-size 8 \
    --use-checkpoint --amp --sample-every 50 --save-every 4 \
    --out-dir ./runs/baseline_check/transformer_regression

python -m method3_transformer.evaluate \
    --checkpoint ./runs/baseline_check/transformer_regression/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 20 --out-dir ./eval_results_baseline_m3
```

**Full training run**, then evaluation, once the baseline check looks right:

```bash
python -m method3_transformer.train_transformer_regression \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --epochs 50 --batch-size 4 --image-size 256 \
    --embed-dim 60 --depths "4,4,4,4" --num-heads 6 --window-size 8 \
    --use-checkpoint --amp --out-dir ./runs/transformer_regression

python -m method3_transformer.evaluate \
    --checkpoint ./runs/transformer_regression/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./data/generated_masks/scratches ./data/generated_masks/smut \
    --num-samples 100 --out-dir ./eval_results_m3
```

---

## What to check after a baseline check, before committing to a full run

- **Did every command complete without errors?**
- **Sample grids** (in each run's `samples/` folder) — any movement toward
  removing damage, even if rough? Total noise/garbage output at this stage
  is worth catching now.
- **`evaluate.py`'s PSNR/SSIM/LPIPS gain will likely be negative or near
  zero** at only 8 epochs — expected, not a failure. What matters is the
  numbers are finite and sensible (no NaNs, no wildly implausible values).
- **Rough per-epoch timing**, printed in each run's `train_log.txt` — use
  this to estimate how long the full 50-epoch runs will actually take.

## Known gaps, stated honestly

- **Method 2's Stage 2 was verified for correctness only against
  randomly-initialized components** (developed in a network-sandboxed
  environment without Hugging Face Hub access) -- the mechanics (gradient
  flow, sampling loop, checkpointing) are confirmed correct, but it has not
  been run against the real pretrained Stable Diffusion checkpoint prior to
  your own first real run.
- None of the three methods include a trained damage-detection component
  applicable to real photographs; damage masks are used only to construct
  synthetic training pairs, and Method 1's translation network receives a
  zero-valued mask at inference on real photos.
- Method 1's face refinement network (present in the original paper) is not
  implemented.
