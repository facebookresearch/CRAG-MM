# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import string
from typing import Any

from PIL import Image

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    RandomAgent is a reference implementation that demonstrates the expected interface
    for the CRAG-MM benchmark.

    This agent returns random strings as responses and serves as a baseline for
    understanding the agent interface. In a real implementation, you would replace this
    with a model that generates meaningful responses based on the query, images, and
    conversation history.

    The agent interface is designed to work with the CRAG-MM evaluation framework,
    which evaluates agents on both single-turn and multi-turn conversation tasks.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the RandomAgent.
        """
        super().__init__()
        print("Initializing RandomAgent - reference implementation")

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling
        batch_generate_response.

        The evaluator uses this value to determine how many queries to send in each
        batch. Valid values are integers between 1 and 16.

        Returns:
            int: The batch size, indicating how many queries should be processed
                 together in a single batch.
        """
        # Fixed batch size of 16 for this reference implementation
        # You may adjust this based on your agent's capabilities
        return 16

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[list[dict[str, Any]]],
        rag_context_list: list[str],
        **kwargs,
    ) -> list[str]:
        """
        Generate responses for a batch of queries.

        This is the main method called by the evaluator. It processes multiple
        queries in parallel for efficiency. For multi-turn conversations,
        the message_histories parameter contains the conversation so far.

        Args:
            queries (list[str]): List of user questions or prompts.
            images (list[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (list[list[dict[str, Any]]]): List of conversation
                histories, one per query. Each history is a list of message dictionaries
                with 'role' and 'content' keys in the following format:

                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the
                  format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
            rag_context_list (list[str]): Additional prompts that include the retrieved
                image search and/or web search results

        Returns:
            list[str]: List of generated responses, one per input query.
        """
        # For illustration, this random agent just generates random strings
        # A real implementation would:
        # 1. Use self.search_pipeline to retrieve relevant information if needed
        # 2. Process each query along with its corresponding image (if available)
        # 3. Consider the conversation history for multi-turn tasks
        # 4. Generate meaningful responses based on retrieved information

        return [
            "".join(
                random.choice(string.ascii_letters + " ")
                for _ in range(random.randint(2, 16))
            )
            for _ in range(len(queries))
        ]
