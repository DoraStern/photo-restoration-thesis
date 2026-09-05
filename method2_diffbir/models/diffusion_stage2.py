"""
Method 2, Stage 2: the frozen Stable Diffusion backbone + the trainable
ControlNet-style conditioning adapter that steers it using Stage 1's
output.

Uses HuggingFace's `diffusers` library directly for the frozen components
(AutoencoderKL, UNet2DConditionModel, CLIP text encoder) and for
ControlNetModel itself, rather than reimplementing any of this from
scratch -- these are large, well-tested public components, and the actual
"method" being recreated here is how they're combined and what gets
trained on top, not the internals of Stable Diffusion itself.

What's frozen vs. trained:
    Frozen:  AutoencoderKL (VAE), UNet2DConditionModel, CLIP text encoder
    Trained: ControlNetModel (initialized from the frozen UNet's encoder
             half via ControlNetModel.from_unet(), then trained from there)

Requires an internet connection to Hugging Face Hub the first time you
run this (to download the pretrained checkpoint) -- this needs to happen
on Kaggle or another environment with unrestricted internet access, not
in a network-sandboxed environment.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# Note: the older "runwayml/stable-diffusion-v1-5" repo was taken down by
# RunwayML; this is the current official re-upload/mirror. If this ID ever
# moves again, any of the community archival mirrors (search "stable
# diffusion v1.5 huggingface mirror") should work as a drop-in replacement
# -- the checkpoint contents are identical, only the repo location differs.


def load_frozen_sd_components(model_id: str, device: torch.device, dtype=torch.float32):
    """Loads VAE, UNet, text encoder, and tokenizer from a pretrained SD
    checkpoint, freezes all of them (no gradients, eval mode), and returns
    them along with a DDPM noise scheduler matching the checkpoint's
    training configuration."""
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    for module in (vae, unet, text_encoder):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    return vae, unet, text_encoder, tokenizer, noise_scheduler


def build_controlnet(unet: UNet2DConditionModel) -> ControlNetModel:
    """Builds the trainable ControlNet-style adapter, initialized from the
    frozen UNet's own encoder weights (standard ControlNet initialization
    -- starts as a near-copy of part of the frozen network, then diverges
    as it trains). conditioning_channels=3 because the hint we feed it is
    Stage 1's restored output as a plain RGB image, not a pre-encoded
    latent -- ControlNetModel has its own small conv stem that embeds the
    raw image internally."""
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)
    return controlnet


def get_empty_prompt_embedding(text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, device: torch.device):
    """This project isn't text-conditioned -- there's no prompt to guide
    restoration with -- but UNet2DConditionModel's cross-attention layers
    still expect an encoder_hidden_states tensor of the right shape. The
    standard approach (same one used by unconditional/conditioning-only
    ControlNet applications) is to encode a fixed empty string once and
    reuse that same embedding for every sample."""
    with torch.no_grad():
        tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").to(device)
        embedding = text_encoder(tokens.input_ids)[0]
    return embedding


def encode_to_latent(vae: AutoencoderKL, image: torch.Tensor) -> torch.Tensor:
    """Encodes an image tensor (expected in [-1, 1], matching the rest of
    this project's normalization convention) into the VAE's latent space,
    applying the scaling factor the way Stable Diffusion's own training
    pipeline does."""
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
    return latent


def decode_from_latent(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_to_latent -- turns a latent back into an image in
    [-1, 1]."""
    with torch.no_grad():
        image = vae.decode(latent / vae.config.scaling_factor).sample
    return image
