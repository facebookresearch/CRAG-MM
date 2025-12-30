"""
Dataset utility functions for CRAG-MM evaluation.

This module provides utilities for dataset preparation and processing,
including domain mapping, text cleaning, and feature vocabulary extraction.
"""

import re
from typing import List


def prepare_feature_vocabularies(dataset_split):
    """Extract feature vocabularies for category encoding from dataset.

    These vocabularies allow conversion between integer indices and string labels.

    Args:
        dataset_split: The dataset split to extract vocabularies from.

    Returns:
        Dictionary containing feature vocabularies for domain, query_category,
        dynamism, and image_quality.
    """
    turns_feature = dataset_split.features["turns"]
    return {
        "domain": turns_feature["domain"],
        "query_category": turns_feature["query_category"],
        "dynamism": turns_feature["dynamism"],
        "image_quality": turns_feature["image_quality"],
    }


def convert_domain_indices_to_names(
    domain_indices: List[int], feature_vocabularies: dict
) -> List[str]:
    """Convert domain integer indices to string names using feature vocabularies.

    Args:
        domain_indices: List of integer domain indices
        feature_vocabularies: Dictionary containing domain feature mappings

    Returns:
        List of domain name strings
    """
    domain_feature = feature_vocabularies["domain"].feature
    domain_names = []

    for domain_idx in domain_indices:
        if isinstance(domain_idx, list) and len(domain_idx) > 0:
            # Handle case where domain is a list (take first element)
            domain_idx = domain_idx[0]

        if domain_idx is not None and 0 <= domain_idx < len(domain_feature.names):
            domain_names.append(domain_feature.names[domain_idx])
        else:
            domain_names.append("other")  # Default fallback

    return domain_names


def clean_text(text: str) -> str:
    """Clean text by lowercasing and removing all non-alphanumeric characters.

    Args:
        text: The text to clean.

    Returns:
        Cleaned text with only lowercase alphanumeric characters and spaces.
    """
    # Lowercase and remove all non-alphanumeric characters (including apostrophes)
    return re.sub(r"[^a-z0-9\s]", "", text.lower())
