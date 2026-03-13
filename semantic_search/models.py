"""SigLIP 2 model loading and encoding.

A single ModelManager instance is created per process and reused for both
indexing and search — the model is only loaded once.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from huggingface_hub.utils import LocalEntryNotFoundError
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google/siglip2-base-patch16-224"


def resolve_device(device: str) -> str:
    """Resolve "auto" to the best available device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class ModelManager:
    """Loads a SigLIP 2 model once and exposes encode_image / encode_text.

    Both methods return L2-normalised float32 numpy arrays of shape (N, D),
    ready to store in or compare against LanceDB.
    """

    def __init__(
        self,
        model_variant: str = _DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self._device = torch.device(resolve_device(device))
        logger.info("Loading %s on %s ...", model_variant, self._device)

        try:
            self._processor = AutoProcessor.from_pretrained(model_variant, local_files_only=True)
            self._model = (
                AutoModel.from_pretrained(model_variant, local_files_only=True)
                .to(self._device)
                .eval()
            )
        except (LocalEntryNotFoundError, OSError):
            logger.info("Model not cached — downloading from Hugging Face...")
            self._processor = AutoProcessor.from_pretrained(model_variant)
            self._model = (
                AutoModel.from_pretrained(model_variant)
                .to(self._device)
                .eval()
            )

        # Detect output dimension with a single dummy forward pass.
        self.embedding_dim: int = self._detect_dim()
        logger.info(
            "Model ready — embedding dim: %d, device: %s",
            self.embedding_dim,
            self._device,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_tensor(output) -> torch.Tensor:
        """Extract a plain tensor from a model output.

        Newer versions of transformers may return a BaseModelOutputWithPooling
        object rather than a bare tensor from get_image/text_features.
        """
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]  # CLS token
        raise ValueError(f"Cannot extract feature tensor from {type(output)}")

    def _detect_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224))
        inputs = self._processor(images=[dummy], return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._as_tensor(self._model.get_image_features(**inputs))
        return int(feats.shape[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL Images.

        Args:
            images: list of RGB PIL Images (any size — the processor resizes them).

        Returns:
            Float32 array of shape (N, D), L2-normalised.
        """
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._as_tensor(self._model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats.cpu().float().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of text strings.

        Args:
            texts: list of query strings.

        Returns:
            Float32 array of shape (N, D), L2-normalised.
        """
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self._device)
        with torch.no_grad():
            feats = self._as_tensor(self._model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats.cpu().float().numpy()
