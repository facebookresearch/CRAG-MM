# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CRAG-MM Dataset Processing Module

This module provides tools for processing and batching the CRAG-MM dataset,
which contains multi-turn conversations with images and text.

The main components are:
1. CRAGTurnBatchIterator - Processes dataset into batches of conversation turns
2. ImageLoader - Handles image loading from various sources
3. Main execution code for debugging and demonstration
"""

import random
from typing import Any

import tqdm
from datasets import Dataset, load_dataset
from loguru import logger
from PIL import Image
from utils import download_image_url, parse_jsonl_to_dict


class ImageLoader:
    """Handles loading and caching of images from various sources."""

    @staticmethod
    def load_image(conversation_data: dict[str, Any]) -> Image.Image:
        """
        Load image from conversation data, downloading if necessary.

        Args:
            conversation_data: Dictionary containing image data or URL

        Returns:
            PIL Image object

        Notes:
            - Either 'image' or 'image_url' will be provided in the dataset
            - When the actual image cannot be included, only the image_url is available
        """
        image = conversation_data.get("image")
        image_url = conversation_data.get("image_url")

        if image is None and image_url:
            # Download image from URL (with local caching)
            image_local_path = download_image_url(image_url)
            image = Image.open(image_local_path)

        return image


class CRAGTurnBatchIterator:
    """
    Processes CRAG-MM dataset into batches of conversation turns.

    This class handles the complex structure of the CRAG dataset, which contains
    multi-turn conversations with images, and converts it into a format suitable
    for model training or evaluation.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        full_query_file: str = None,
        shuffle: bool = False,
        sessions_to_skip: list = [],
        sessions_to_update: dict = {},
    ):
        """
        Initialize the batcher with dataset and parameters.

        Args:
            dataset: HuggingFace dataset containing CRAG conversations
            batch_size: Number of conversation turns to include in each batch
            shuffle: Whether to shuffle the conversation order
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.sessions_to_skip = sessions_to_skip
        self.sessions_to_update = sessions_to_update
        self.full_query_map = parse_jsonl_to_dict(full_query_file)

        if self.shuffle:
            random.shuffle(self.indices)

    def _extract_turn_data(
        self,
        conversation: dict[str, Any],
        turn_idx: int,
        turn: dict[str, Any],
        session_id: str,
        image: Image.Image,
        image_url: str,
        image_quality: str,
        answer_lookup: dict[str, str],
        total_turn_count: int,
    ) -> dict[str, Any]:
        """
        Extract data for a single conversation turn.

        Args:
            conversation: Full conversation data
            turn_idx: Index of the current turn
            turn: Data for the current turn
            session_id: ID of the conversation session
            image: Image associated with the conversation
            image_url: URL of the image
            image_quality: Quality rating of the image
            answer_lookup: Dictionary mapping interaction IDs to answers
            total_turn_count: Total number of turns in the conversation

        Returns:
            Dictionary with processed turn data
        """
        # Extract basic turn information
        interaction_id = turn["interaction_id"]
        query = turn["query"]
        query_category = turn["query_category"]
        domain = turn["domain"]
        dynamism = turn["dynamism"]

        # Get ground truth answer
        answer = answer_lookup.get(interaction_id, False)
        assert (
            answer is not False
        ), f"No answer found for interaction_id: {interaction_id}"

        # Extract conversation history up to this point
        conversation_history = []
        answer_history = []
        if turn_idx > 0:
            conversation_history = conversation["turns"][:turn_idx]
            answer_history = [
                answer_lookup.get(a["interaction_id"], False)
                for a in conversation_history
            ]
            assert all(answer_history), (
                f"No answer found for turn history of interaction_id: {interaction_id}."
                f" session_id: {session_id}"
            )

        # Resize egocentric frames
        if image_url is None or len(image_url) == 0:
            image = image.resize((960, 1280))

        # Return structured turn data

        return {
            "session_id": session_id,
            "interaction_id": interaction_id,
            "turn_idx": turn_idx,
            "image": image,
            "image_url": image_url,
            "image_quality": image_quality,
            "full_query": self.full_query_map.get(interaction_id, ""),
            "query": query,
            "answer": answer,
            "query_category": query_category,
            "domain": domain,
            "dynamism": dynamism,
            "conversation_history": conversation_history,
            "answer_history": answer_history,
            "total_turn_count": total_turn_count,
        }

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """
        Collate individual turn data into batch format.

        Args:
            batch: List of dictionaries with turn data

        Returns:
            Dictionary with batched data
        """
        # Initialize lists for all fields
        batch_data = {
            "session_ids": [],
            "interaction_ids": [],
            "turn_idxs": [],
            "images": [],
            "image_urls": [],
            "image_qualities": [],
            "full_queries": [],
            "queries": [],
            "answers": [],
            "query_categories": [],
            "domains": [],
            "dynamisms": [],
            "conversation_histories": [],
            "answer_histories": [],
            "total_turn_counts": [],
        }

        # Collect data from each item
        for item in batch:
            batch_data["session_ids"].append(item["session_id"])
            batch_data["interaction_ids"].append(item["interaction_id"])
            batch_data["turn_idxs"].append(item["turn_idx"])

            batch_data["images"].append(item["image"])
            batch_data["image_urls"].append(item["image_url"])
            batch_data["image_qualities"].append(item["image_quality"])

            batch_data["full_queries"].append(item["full_query"])
            batch_data["queries"].append(item["query"])
            batch_data["answers"].append(item["answer"])

            batch_data["query_categories"].append(item["query_category"])
            batch_data["domains"].append(item["domain"])
            batch_data["dynamisms"].append(item["dynamism"])

            batch_data["conversation_histories"].append(item["conversation_history"])
            batch_data["answer_histories"].append(item["answer_history"])

            batch_data["total_turn_counts"].append(item["total_turn_count"])

        return batch_data

    def __iter__(self):
        """
        Iterate through the dataset and yield batches of turns, ensuring that
        turn N+1 for any conversation is only in a strictly later batch
        than turn N from the same conversation.

        This is critical for the correct batching strategy for the multi-turn
        conversations, but should also work for the single-turn conversations.
        """
        from collections import deque

        # For each conversation, track the next turn to produce:
        # next_turn_idx[i] = k means we've consumed turns [0..k-1].
        next_turn_idx = [0] * len(self.dataset)

        # Initialize the queue of conversation indices that have turns left
        queue = deque(self.indices)

        # A cache for conversation data, images, and answers
        self.conversation_cache = {}  # conv_id -> conversation dict
        self.answer_lookup_cache = {}  # conv_id -> {interaction_id -> ans_full}
        self.image_cache = {}  # conv_id -> loaded PIL image

        # We'll accumulate turn data in 'batch' each iteration
        batch = []

        while queue:
            current_convs = []
            # Pop conversation IDs from the queue up to batch_size
            while queue and len(current_convs) < self.batch_size:
                conv_id = queue.popleft()
                current_convs.append(conv_id)

            # Process exactly one turn per conversation in current_convs
            for conv_id in current_convs:
                if self.dataset[conv_id]["session_id"] in self.sessions_to_skip:
                    logger.warning(
                        "Skipping session {}", self.dataset[conv_id]["session_id"]
                    )
                    continue

                conv_data = self.dataset[conv_id]
                if conv_data["session_id"] in self.sessions_to_update:
                    conv_data.update(self.sessions_to_update[conv_data["session_id"]])

                # ---------------------------
                # 1) LAZY-LOAD IF NECESSARY
                # ---------------------------
                if conv_id not in self.conversation_cache:
                    # Load from dataset once
                    self.conversation_cache[conv_id] = conv_data

                    # Build answer lookup
                    # conv_data for round 2 dataset is of the form
                    # {
                    #     "answers": {
                    #         "ans_full": [...],
                    #         "interaction_id": [...],
                    #     },
                    #     ...
                    # }
                    # conv_data for round 1 dataset is of the form
                    # {
                    #     "answers": [
                    #         {
                    #             "interaction_id": ...,
                    #             "ans_full": ...,
                    #         },
                    #         ...
                    #     ],
                    #     ...
                    # }
                    if isinstance(conv_data, dict):
                        answers = []
                        for idx in range(len(conv_data["answers"]["interaction_id"])):
                            answers.append(
                                {
                                    "interaction_id": conv_data["answers"][
                                        "interaction_id"
                                    ][idx],
                                    "ans_full": conv_data["answers"]["ans_full"][idx],
                                }
                            )
                        conv_data["answers"] = answers
                    ans_lookup = {
                        a["interaction_id"]: a["ans_full"] for a in conv_data["answers"]
                    }
                    self.answer_lookup_cache[conv_id] = ans_lookup

                    # Load and cache the image
                    self.image_cache[conv_id] = ImageLoader.load_image(conv_data)

                # -------------------------
                # 2) FETCH FROM THE CACHE
                # -------------------------
                conversation = self.conversation_cache[conv_id]
                answer_lookup = self.answer_lookup_cache[conv_id]
                image = self.image_cache[conv_id]

                # Identify which turn we need
                # conv_data for round 2 dataset is of the form
                # {
                #     "turns": {
                #         "interaction_id": [...],
                #         "query": [...],
                #         ...
                #     },
                #     ...
                # }
                # conv_data for round 1 dataset is of the form
                # {
                #     "turns": [
                #         {
                #             "interaction_id": ...,
                #             "query": ...,
                #         },
                #         ...
                #     ],
                #     ...
                # }
                if isinstance(conversation["turns"], dict):
                    turns = []
                    for idx in range(len(conversation["turns"]["interaction_id"])):
                        _sample = {}
                        for k, v in conversation["turns"].items():
                            _sample[k] = v[idx]
                        turns.append(_sample)
                    conversation["turns"] = turns

                turn_idx = next_turn_idx[conv_id]
                total_turn_count = len(conversation["turns"])
                turn = conversation["turns"][turn_idx]

                image_url = conversation.get("image_url", None)
                image_quality = conversation.get("image_quality", None)

                # Build the single turn's data
                turn_data = self._extract_turn_data(
                    conversation=conversation,
                    turn_idx=turn_idx,
                    turn=turn,
                    session_id=conversation["session_id"],
                    image=image,
                    image_url=image_url,
                    image_quality=image_quality,
                    answer_lookup=answer_lookup,
                    total_turn_count=total_turn_count,
                )
                batch.append(turn_data)

                # ------------------------
                # 3) UPDATE TURN POINTER
                # ------------------------
                next_turn_idx[conv_id] += 1

                # If that conversation still has turns left, re-append it to the (front
                # of the) queue
                if next_turn_idx[conv_id] < total_turn_count:
                    queue.appendleft(conv_id)
                    # note: appending to the left helps keep the cache size in check
                else:
                    # No more turns in this conversation => remove from cache
                    del self.conversation_cache[conv_id]
                    del self.answer_lookup_cache[conv_id]
                    del self.image_cache[conv_id]

            # Yield the entire batch as one chunk
            yield self._collate_batch(batch)
            batch = []

        # If there's anything left in batch, yield it
        if batch:
            yield self._collate_batch(batch)


def main():
    """
    Main function for demonstration and debugging.

    Loads the CRAG-MM dataset, processes it through the batcher,
    and prints each batch to verify the implementation.
    """
    # Load dataset
    print("Loading CRAG-MM dataset...")
    from utils import ensure_crag_cache_dir_is_configured

    CACHE_DIR = ensure_crag_cache_dir_is_configured()

    dataset = load_dataset(
        "crag-mm-2025/crag-mm-single-turn-public",
        revision="v0.1.2",
        cache_dir=CACHE_DIR,
    )
    dataset_split = dataset["public_test"]

    # Create batcher
    print(f"Creating batcher for {len(dataset_split)} conversations...")
    batch_size = 16
    full_query_file = "full_query/single_turn_full_query.jsonl"
    crag_turn_batch_iterator = CRAGTurnBatchIterator(
        dataset=dataset_split,
        batch_size=batch_size,
        full_query_file=full_query_file,
        shuffle=True,
    )
    # Process batches
    print(f"Processing dataset in batches of {batch_size}...")
    for batch in tqdm.tqdm(crag_turn_batch_iterator):
        # Just print batch size for demonstration
        num_examples = len(batch["session_ids"])
        print(f"Batch with {num_examples} examples")
        print(f"Cache size: {len(crag_turn_batch_iterator.conversation_cache)}")
        # Uncomment to print full batch details:
        # print(batch)


if __name__ == "__main__":
    main()
