# Digital Photograph Restoration — Thesis Comparison

Three restoration methods, compared on the same shared data: a GAN + VAE
latent-translation method, DiffBIR-style diffusion restoration (Stage 1
built so far), and a transformer regression method (Swin/SwinIR-style).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── common/                        # Shared, method-agnostic code
│   ├── blend.py                   #   damage compositing (screen/multiply) --
│   │                               #   single source of truth, used by the CLI
│   │                               #   tool AND both dataset classes below
│   ├── degraded_pair_dataset.py   #   used by Method 2 AND Method 3 (identical)
│   └── real_photo_dataset.py      #   used by Method 1 (domain A)
│
├── data_tools/                    # One-time data preparation scripts
│   ├── scrapers/
│   │   ├── loc_scraper.py         #   Library of Congress real old photos
│   │   └── dpla_scraper.py        #   Digital Public Library of America
│   └── damage_generation/
│       ├── generate_synthetic_only.py   # FilmDamageSimulator mask generator
│       └── composite_damage.py           # standalone compositing CLI tool
│
├── method1_gan_vae/                # GAN + VAE latent translation
│   ├── models/
│   │   ├── vae_blocks.py           #   NOTE: renamed from blocks.py -- Method 2
│   │   │                           #   has its own different blocks.py; keeping
│   │   │                           #   them distinctly named avoids a real
│   │   │                           #   naming collision that existed earlier
│   │   ├── vae.py                  #   DomainVAE (shared class, used for both
│   │   │                           #   VAE1 and VAE2, trained separately)
│   │   └── translation_net.py      #   LatentTranslationNet + LatentDiscriminator
│   ├── data/
│   │   └── vae2_pair_dataset.py    #   clean+degraded+mask triples (VAE2-specific)
│   ├── train_vae_domain_a.py       #   Stage 1: VAE1 on real old photos
│   ├── train_vae_domain_b.py       #   Stage 2: VAE2 on clean + synthetic damage
│   └── train_translation_net.py    #   Stage 3: the actual repair mechanism
│
├── method2_diffbir/                 # DiffBIR-style (Stage 1 built; Stage 2
│   │                                 # -- frozen diffusion backbone + ControlNet-
│   │                                 # style adapter -- not yet implemented)
│   ├── models/
│   │   ├── unet_blocks.py           #   NOTE: renamed from blocks.py, see above
│   │   └── restoration_net.py       #   RestorationUNet (Stage 1 regression net)
│   ├── train_stage1_restoration.py
│   └── evaluate.py                  #   PSNR/SSIM evaluation for this method's checkpoints
│
├── method3_transformer/             # Transformer regression (Swin/SwinIR-style)
│   ├── models/
│   │   └── transformer_restoration.py
│   └── train_transformer_regression.py
│
├── notebooks/                       # Kaggle notebooks (see "Cloud training" below)
│   ├── method1_gan_vae_kaggle.ipynb
│   ├── method2_diffbir_kaggle.ipynb
│   ├── method3_transformer_kaggle.ipynb
│   └── git_clone_template_kaggle.ipynb   # recommended NEW workflow, see below
│
└── data/                            # NOT committed to git -- see .gitignore.
    ├── real_old_photos/             #   from data_tools/scrapers/*.py
    │   ├── images/
    │   └── manifest.csv
    ├── voc2012/                     #   clean photos (VOCdevkit/VOC2012/JPEGImages)
    └── generated_masks/             #   from generate_synthetic_only.py
```

## Why this structure

**One `data/` folder, shared by all three methods.** All three methods need
overlapping inputs: clean photos and damage masks (Methods 2 and 3 need
exactly the same two things; Method 1 additionally needs real old photos).
Rather than each method downloading/generating its own copy, run the
scraper and mask generator _once_, into `data/`, and point every method's
`--clean-dir`/`--masks-dir`/`--real-photo-dir` at those same shared
folders. This is also why `common/degraded_pair_dataset.py` is a single
shared file rather than being duplicated into `method2_diffbir/` and
`method3_transformer/` separately — one dataset class, one fix location if
anything about it ever needs to change.

**`common/blend.py` exists because of a real bug pattern from earlier in
this project.** The screen/multiply compositing logic used to be
duplicated in three places (the standalone CLI tool and both dataset
classes), which meant a fix had to be applied three times. It's now one
function, imported everywhere it's used.

**`vae_blocks.py` / `unet_blocks.py` instead of two files both named
`blocks.py`.** Method 1's VAE architecture and Method 2's U-Net
architecture each had their own `blocks.py` with completely different
contents (`ResidualBlock`/`DownsampleBlock`/`UpsampleBlock` vs.
`ConvBlock`/`DownBlock`/`UpBlock`). If these ever ended up in a shared
folder or a flattened import path, one would silently shadow the other.
Distinct names remove that risk entirely rather than relying on directory
structure alone to keep them apart.

## Setup

```bash
pip install -r requirements.txt
```

## One-time shared data preparation

Run these once, before touching any of the three methods:

```bash
# Real old photos (Method 1 only) -- see the scraper's own docstring for
# the --categories/--weights balancing options
pip install gdown
python download_gdrive_dataset.py --file-url "https://drive.google.com/file/d/1knqFN2K026XfFg53uKQjJlcw0yEXcpqm/view?usp=sharing" \
    --out-dir ./data/real_old_photos


# Clean photos (all three methods) -- VOC2012, auto-downloaded via torchvision
python -c "import torchvision.datasets as d; d.VOCDetection(root='./data/voc2012', year='2012', image_set='train', download=True)"

# Damage masks (all three methods)
cd data_tools/damage_generation
git clone --depth 1 https://github.com/daniela997/FilmDamageSimulator.git
cp generate_synthetic_only.py FilmDamageSimulator/damage_generator/
cd FilmDamageSimulator/damage_generator

python generate_synthetic_only.py --types scratches --height 256 --width 256 --n 3000 --out-dir ../../../../data/generated_masks
python generate_synthetic_only.py --types smut --height 256 --width 256 --n 3000 --out-dir ../../../../data/generated_masks
python generate_synthetic_only.py --types dirt --height 256 --width 256 --n 3000 --out-dir ../../../../data/generated_masks
python generate_synthetic_only.py --types spots --height 256 --width 256 --n 3000 --out-dir ../../../../data/generated_masks
python generate_synthetic_only.py --types sprinkles --height 256 --width 256 --n 3000 --out-dir ../../../../data/generated_masks
cd ../../../..
```

## Running each method

All commands run from the **repo root**, using `python -m` so the shared
`common/` package resolves correctly:

```bash
# Method 1 -- three stages, in order
python -m method1_gan_vae.train_vae_domain_a \
    --data-root ./data/real_old_photos --epochs 50 --batch-size 16 --image-size 256 \
    --amp --out-dir ./runs/vae_domain_a

python -m method1_gan_vae.train_vae_domain_b \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages --masks-dir ./data/generated_masks \
    --epochs 50 --batch-size 16 --image-size 256 --amp --out-dir ./runs/vae_domain_b

python -m method1_gan_vae.train_translation_net \
    --vae1-checkpoint ./runs/vae_domain_a/checkpoints/<latest>.pt \
    --vae2-checkpoint ./runs/vae_domain_b/checkpoints/<latest>.pt \
    --real-photo-dir ./data/real_old_photos \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages --masks-dir ./data/generated_masks \
    --epochs 50 --batch-size 16 --image-size 256 --out-dir ./runs/translation_net

# Method 2 -- Stage 1 (Stage 2 not yet implemented)
python -m method2_diffbir.train_stage1_restoration \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages --masks-dir ./data/generated_masks \
    --epochs 50 --batch-size 16 --image-size 256 --amp --out-dir ./runs/stage1_restoration

python -m method2_diffbir.evaluate \
    --checkpoint ./runs/stage1_restoration/checkpoints/<latest>.pt \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages --masks-dir ./data/generated_masks \
    --num-samples 100 --out-dir ./eval_results

# Method 3 -- transformer regression
python -m method3_transformer.train_transformer_regression \
    --clean-dir ./data/voc2012/VOCdevkit/VOC2012/JPEGImages --masks-dir ./data/generated_masks \
    --epochs 50 --batch-size 4 --image-size 256 \
    --embed-dim 60 --depths "4,4,4,4" --num-heads 6 --window-size 8 \
    --use-checkpoint --amp --out-dir ./runs/transformer_regression
```

All three training scripts share the same conventions: `--steps-per-epoch`
to shorten an epoch for quick tests, `--amp` for mixed precision,
`--resume <checkpoint>` to continue, `--log-every` for the timing
diagnostics, and a `train_log.txt` written inside each run's `--out-dir`.

## Cloud training (Kaggle)

The `notebooks/` folder has three tested, working notebooks (one per
method), each self-contained via `%%writefile` cells that recreate the
project's files directly in the Kaggle session.

**Once this repo is pushed to GitHub, consider switching to
`git clone` instead** — `notebooks/git_clone_template_kaggle.ipynb` shows
this pattern. It's meaningfully more robust: with `%%writefile`-based
notebooks, updating any file means editing that specific cell and
remembering to re-run it before your next training command, which is
exactly the failure mode that came up repeatedly during this project
(stale files/variables from unreachable cells). With `git clone` +
`git pull`, updating the whole project is one command, and there's no way
to accidentally run against a stale version of one file while others are
current.

## Held-out evaluation set

Before running your final cross-method comparison, set aside a slice of
clean images (100-200+) that is **never** used in training for _any_ of
the three methods — not VOC2012 training, not mask compositing, not
VAE1/VAE2/translation-network training. This is what all three methods'
final PSNR/SSIM/LPIPS comparison should run against. Decide on this split
once, early, and keep it fixed for the rest of the project.
