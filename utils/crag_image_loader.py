# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DO NOT MODIFY THIS FILE.

This file contains ImageLoader class which can be used to download and cache images.
During evaluation, the cache is prepopulated with all available images and everything is
loaded from cache. Internet access is disabled during evaluation and using image URLs
that are not part of the provided dataset would lead to an error (All `requests.get`
calls would fail).
"""

import os
from hashlib import sha256
from io import BytesIO

import requests
from PIL import Image

CACHE_DIR = os.getenv(
    "CRAG_IMAGE_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/crag/", "image_cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)


class ImageLoader:
    def __init__(self, url: str):
        self.url = url

    def _get_cache_filename(self):
        file_ext = self.url.split(".")[-1].lower()
        return os.path.join(
            CACHE_DIR, sha256(self.url.encode()).hexdigest() + "." + file_ext
        )

    def _save_image_to_cache(self, image: Image.Image):
        image.save(self._get_cache_filename())

    def _load_image_from_cache(self):
        return Image.open(self._get_cache_filename())

    def _image_cache_exists(self):
        return os.path.exists(self._get_cache_filename())

    def download_image(self):
        headers = {"User-Agent": "CRAGBot/v0.0.1"}
        response = requests.get(self.url, stream=True, timeout=10, headers=headers)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            raise Exception(
                f"Failed to download image from {self.url}, status code:"
                f" {response.status_code}"
            )

    def get_image(self):
        if self._image_cache_exists():
            return self._load_image_from_cache()
        else:
            image = self.download_image()
            self._save_image_to_cache(image)
            return image
