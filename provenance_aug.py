"""
Phase 7D: provenance randomisation.

The Phase 7C domain probe reached 0.903 accuracy telling FER+ crops from ExpW
crops, against a chance baseline of 0.50. That is a shortcut: disgust is 90%
ExpW-sourced, so "this image came from ExpW" is a usable proxy for "disgust",
and a model is free to learn the corpus instead of the face.

WHAT THE SIGNATURE ACTUALLY IS
------------------------------
It is not brightness or contrast - those were measured and are nearly identical
(mean pixel 128.4 vs 126.6, contrast 54.6 vs 51.5). It is RESAMPLING HISTORY:

  * a FER2013 image is natively 48x48. It has never been downscaled; its
    high-frequency content is whatever the 2013 collection pipeline left.
  * an ExpW crop starts as a face of median ~96px inside a web JPEG and is
    reduced to 48x48 by INTER_AREA, which leaves a different spectral
    fingerprint and different JPEG block structure.

A network reads that fingerprint easily. Normalising brightness cannot remove
it, because it lives in the texture statistics.

THE FIX
-------
Put every image through a randomised resampling chain, so that no crop retains
its native history and the fingerprint becomes noise rather than signal. The
chain is applied to BOTH corpora with the same parameter distribution - applying
it to one side only would replace one signature with another.

WHY THESE OPERATIONS ARE REALISTIC, NOT ARBITRARY
-------------------------------------------------
This is the exact channel a live webcam frame already travels:

    camera -> canvas -> JPEG at quality 0.72 (frontend/src/hooks/useLiveCamera.ts)
           -> POST -> cv2.imdecode -> BlazeFace crop
           -> to_fer48(): INTER_AREA down to 48x48

so lossy JPEG and an unknown-ratio downscale are deployment conditions, not
distortions invented for the probe. Randomising the ratio and the quality makes
the model robust to the whole family of resampling chains instead of memorising
one.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Nothing that could change which expression a face is showing: no geometric
warping, no aspect distortion, no rotation, no occlusion, no contrast inversion,
no scale so small the facial action is destroyed. Resolution is degraded, never
below half, and the expression survives - that is the difference between
degrading a channel and editing a face.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- resampling chain parameters -----------------------------------------
# Effective source resolution, as a fraction of 48px. 1.00 leaves the size
# alone; 0.50 simulates a face that reached us through a 24px channel. Below
# half, facial actions genuinely start to disappear, so that is the floor.
SCALE_MIN = 0.50
SCALE_MAX = 1.00
# JPEG quality. The live client already encodes at 0.72, so 72 sits inside this
# range by construction; the tails cover better and worse connections.
JPEG_Q_MIN = 30
JPEG_Q_MAX = 95
# Mild interpolation variation: which kernel resampled this image is itself part
# of the provenance fingerprint, so it is randomised too. INTER_NEAREST is
# excluded - it produces blocky aliasing no real pipeline would emit.
DOWN_INTERP = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
UP_INTERP = (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA)

FER_SIZE = 48


class ProvenanceRandomiser:
    """Callable applied to a (48,48) uint8 crop before any tensor transform.

    Counts its own invocations. The count is asserted against the number of
    samples the model actually consumed, so "the augmentation is enabled" is a
    measured fact rather than a claim about the config.
    """

    def __init__(self, scale=(SCALE_MIN, SCALE_MAX),
                 quality=(JPEG_Q_MIN, JPEG_Q_MAX), seed: int | None = None):
        self.scale = scale
        self.quality = quality
        self.calls = 0
        self._seed = seed
        self._rng: np.random.Generator | None = None

    def _gen(self) -> np.random.Generator:
        # Built lazily and per worker: DataLoader workers are forked/spawned
        # copies, and a shared Generator would hand every worker the same
        # parameter stream.
        if self._rng is None:
            import os
            seed = self._seed if self._seed is not None else 0
            self._rng = np.random.default_rng(seed + os.getpid())
        return self._rng

    def __call__(self, img48: np.ndarray) -> np.ndarray:
        rng = self._gen()
        self.calls += 1
        img = img48

        # 1. random downscale, with a random kernel
        s = float(rng.uniform(*self.scale))
        side = max(8, int(round(FER_SIZE * s)))
        if side != FER_SIZE:
            img = cv2.resize(img, (side, side),
                             interpolation=DOWN_INTERP[rng.integers(len(DOWN_INTERP))])

        # 2. JPEG recompression at the reduced size - the same lossy channel a
        #    webcam frame travels, at a randomised quality
        q = int(rng.integers(self.quality[0], self.quality[1] + 1))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                img = decoded

        # 3. back up to 48x48, with a random kernel
        if img.shape[0] != FER_SIZE or img.shape[1] != FER_SIZE:
            img = cv2.resize(img, (FER_SIZE, FER_SIZE),
                             interpolation=UP_INTERP[rng.integers(len(UP_INTERP))])

        return np.ascontiguousarray(img, dtype=np.uint8)

    def describe(self) -> dict:
        return {
            "always_applied": True,
            "downscale_fraction_of_48px": {"min": self.scale[0], "max": self.scale[1],
                                           "distribution": "uniform"},
            "resulting_intermediate_side_px": {
                "min": int(round(FER_SIZE * self.scale[0])),
                "max": int(round(FER_SIZE * self.scale[1]))},
            "jpeg_quality": {"min": self.quality[0], "max": self.quality[1],
                             "distribution": "uniform integer",
                             "note": "the live client encodes at 0.72, inside this range"},
            "downscale_interpolation": ["INTER_AREA", "INTER_LINEAR", "INTER_CUBIC"],
            "upscale_interpolation": ["INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"],
            "order": "downscale -> JPEG encode/decode at reduced size -> upscale to 48",
            "applied_to": "both corpora, identical parameter distribution",
            "excluded_on_purpose": [
                "geometric warping", "aspect distortion", "rotation", "occlusion",
                "contrast inversion", "any scale below 0.5 of 48px"],
            "expression_preserving": True,
        }


# =========================================================================
# Phase 7E: the content hypothesis.
# =========================================================================
# Provenance randomisation cut the FER+/ExpW texture gap by 65% (Laplacian
# variance 457 -> 162) and the probe still separated the corpora at a median
# 0.786. So the signature is probably not a low-level channel artifact at all.
#
# This transform settles it. A heavy low-pass destroys the entire high-frequency
# band - JPEG blocks, resampling ringing, sensor noise, compression texture,
# every artifact a corpus could be fingerprinted by - while leaving the coarse
# structure that actually encodes an expression: the configuration of brows,
# eyes and mouth, which at 48x48 lives at scales of roughly 10-20 pixels.
#
# If the probe still separates the corpora after this, no augmentation will fix
# it, because what remains is CONTENT: who was photographed, how they were
# framed, and what the two scrapes collected. That is a permanent property of
# the datasets, not something a pipeline can normalise away.
#
# Unlike ProvenanceRandomiser this is DETERMINISTIC - no randomness at all. The
# probe must not be able to exploit parameter variation as an incidental cue,
# and a fixed kernel makes the experiment exactly reproducible.

class ExpressionPreservingBlur:
    """Fixed Gaussian low-pass on the 48x48 crop. Identical for both corpora."""

    def __init__(self, sigma: float = 2.0):
        self.sigma = float(sigma)
        # Kernel wide enough that the Gaussian is not truncated: +-3 sigma.
        k = int(2 * round(3 * self.sigma) + 1)
        self.ksize = max(3, k if k % 2 == 1 else k + 1)
        self.calls = 0

    def __call__(self, img48: np.ndarray) -> np.ndarray:
        self.calls += 1
        out = cv2.GaussianBlur(img48, (self.ksize, self.ksize),
                               sigmaX=self.sigma, sigmaY=self.sigma,
                               borderType=cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(out, dtype=np.uint8)

    def describe(self) -> dict:
        return {
            "always_applied": True,
            "deterministic": True,
            "kernel": "Gaussian",
            "sigma_px": self.sigma,
            "kernel_size_px": self.ksize,
            "border": "BORDER_REPLICATE",
            "image_size_px": FER_SIZE,
            "applied_to": "both corpora, identical procedure",
            "purpose": "remove the entire high-frequency band (JPEG blocks, "
                       "resampling ringing, sensor noise) while retaining the "
                       "coarse brow/eye/mouth configuration that carries the "
                       "expression",
            "randomness": "none - a fixed kernel cannot itself become a cue",
        }
