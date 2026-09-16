# Pametna restauracija fotografija

Repozitorij uz diplomski rad **"Pametna restauracija: usporedba suvremenih metoda za restauraciju fotografija"** (Dora Štern, FERIT Osijek, 2026.). Sadrži Kaggle bilježnice (notebooks) korištene za generiranje sintetičkih oštećenja, obučavanje i evaluaciju tri implementirane metode restauracije starih fotografija:

1. **GAN + VAE** — dva varijacijska autoenkodera (VAE1, VAE2) povezana mrežom za prijenos između latentnih prostora.
2. **Dvofazni difuzijski model** — regresijski modul (faza 1) + ControlNet (faza 2).
3. **Transformerska regresija**.

## Struktura repozitorija

```
photo-restoration-thesis/
├── README.md
└── notebooks/
    ├── 00_generiranje_maski.ipynb
    ├── 01_vae1_training.ipynb
    ├── 02_vae2_training.ipynb
    ├── 03_translation_network_training.ipynb
    ├── 04_diffusion_stage1_training.ipynb
    ├── 05_diffusion_stage2_controlnet_training.ipynb
    ├── 06_transformer_regression_training.ipynb
    ├── 07_evaluation_gan_vae.ipynb
    ├── 08_evaluation_diffusion.ipynb
    └── 09_evaluation_transformer.ipynb
```

## Redoslijed pokretanja

| # | Bilježnica | Opis |
|---|---|---|
| 00 | `generiranje_maski.ipynb` | Generira sintetičke maske oštećenja (ogrebotine, mrlje, točkice, nečistoće) u četirima kategorijama; izlaz je 12000 maski koje se koriste kao ulazni podatak za sve tri metode. |
| 01 | `vae1_training.ipynb` | Obučavanje VAE1 komponente (GAN + VAE metoda). |
| 02 | `vae2_training.ipynb` | Obučavanje VAE2 komponente (GAN + VAE metoda). |
| 03 | `translation_network_training.ipynb` | Obučavanje mreže za prijenos između latentnih prostora VAE1 i VAE2. |
| 04 | `diffusion_stage1_training.ipynb` | Obučavanje regresijskog modula — prva faza difuzijske metode. |
| 05 | `diffusion_stage2_controlnet_training.ipynb` | Obučavanje ControlNet komponente — druga faza difuzijske metode. |
| 06 | `transformer_regression_training.ipynb` | Obučavanje transformerske regresijske mreže. |
| 07 | `evaluation_gan_vae.ipynb` | Učitavanje obučenih VAE1/VAE2/translacijske mreže, spajanje u cjevovod, kvantitativna i kvalitativna evaluacija (PSNR, SSIM, LPIPS, grafovi, mreže primjera). |
| 08 | `evaluation_diffusion.ipynb` | Evaluacija dvofaznog difuzijskog cjevovoda. |
| 09 | `evaluation_transformer.ipynb` | Evaluacija transformerske regresijske metode. |

Bilježnice `01`–`06` moraju se pokrenuti prije odgovarajućih evaluacijskih bilježnica (`07`–`09`). Hiperparametri korišteni pri obučavanju (npr. `--image-disc-lr`, `--kl-weight`, `--guidance-scale`) definirani su kao konfigurabilni parametri na početku svake bilježnice za obučavanje.

## Podaci

Sve tri metode obučavane su na sintetički generiranim oštećenjima primijenjenim na čiste fotografije; jedna od metoda dodatno je obučena i na stvarnim starim fotografijama. *(Ovdje po potrebi dodati izvor/poveznicu na skup čistih fotografija, npr. Library of Congress, ako ga želiš javno navesti.)*

## Okruženje

Sve bilježnice pokretane su na platformi [Kaggle](https://www.kaggle.com/), korištenjem GPU-a dostupnog na toj platformi putem CUDA-e. *(Dopuni: točna verzija Pythona/PyTorcha i popis korištenih biblioteka — vidi napomenu u nastavku.)*
