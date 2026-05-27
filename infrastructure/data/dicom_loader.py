"""DICOM radiograph image parser and loader.

Enables seamless ingestion of medical standard DICOM files and scales raw
pixel intensities into standardized RGB PIL images.
"""

from __future__ import annotations

import numpy as np
import pydicom
from PIL import Image


class DicomImageLoader:
    """Loader utility to load and convert DICOM files into PIL Image objects."""

    @staticmethod
    def load(path: str) -> Image.Image:
        """Read a DICOM file from disk and convert its pixel array to an RGB PIL Image.

        Args:
            path: Absolute or relative path to the DICOM file.

        Returns:
            A PIL Image object in RGB format.
        """
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array.astype(float)

        # Normalize pixel values to the range [0, 255]
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        if max_val > min_val:
            scaled = (pixel_array - min_val) / (max_val - min_val) * 255.0
        else:
            scaled = np.zeros_like(pixel_array)

        # Convert to 8-bit unsigned integer
        scaled_uint8 = np.uint8(scaled)

        # Standardize grayscale array into RGB representation
        return Image.fromarray(scaled_uint8).convert("RGB")
