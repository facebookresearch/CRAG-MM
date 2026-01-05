# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation module for CRAG-MM dataset.

This module provides a modular evaluation framework for the CRAG-MM benchmark,
with components for configuration, dataset utilities, LLM-based judging, and evaluation orchestration.
"""

from evaluation.config import (
    DEFAULT_EVAL_MODEL,
    DEFAULT_NUM_WORKERS,
    MAX_API_RETRIES,
    MAX_BATCH_SIZE,
    MAX_EVAL_TOKENS,
    MAX_RESPONSE_LENGTH_IN_TOKENS,
    MIN_BATCH_SIZE,
    SESSIONS_TO_SKIP,
    SESSIONS_TO_UPDATE,
)
from evaluation.dataset_utils import (
    clean_text,
    convert_domain_indices_to_names,
    prepare_feature_vocabularies,
)
from evaluation.evaluator import CRAGEvaluator
from evaluation.llm_judge import LLMJudge

__all__ = [
    # Evaluator
    "CRAGEvaluator",
    "LLMJudge",
    # Config
    "DEFAULT_EVAL_MODEL",
    "DEFAULT_NUM_WORKERS",
    "MAX_API_RETRIES",
    "MAX_BATCH_SIZE",
    "MAX_EVAL_TOKENS",
    "MAX_RESPONSE_LENGTH_IN_TOKENS",
    "MIN_BATCH_SIZE",
    "SESSIONS_TO_SKIP",
    "SESSIONS_TO_UPDATE",
    # Dataset Utils
    "clean_text",
    "convert_domain_indices_to_names",
    "prepare_feature_vocabularies",
]
