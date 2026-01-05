# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, TypeAlias

from PIL import Image
from utils import resize_image_if_needed

from .prompt_constants import SYSTEM_MULTI_PROMPT, SYSTEM_PROMPT


Message: TypeAlias = dict[str, Any]
Messages: TypeAlias = list[Message]


class BaseAgent:
    """
    BaseAgent is the abstract base class for all CRAG-MM benchmark agents.

    Any agent implementation for the CRAG-MM benchmark should inherit from this class
    and implement the required methods. The agent is responsible for generating
    responses to user queries, potentially using images and conversation history for
    context.

    The CRAG-MM evaluation framework evaluates agents on both single-turn and
    multi-turn conversation tasks.
    """

    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement this method")

    def _build_text_content(self, text: str) -> Any:
        return text

    def _build_image_content(self, image: Image.Image) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def _build_message(self, role: str, content: Any) -> Message:
        return {"role": role, "content": content}

    def _add_message(self, messages: Messages, message: Message) -> None:
        messages.append(message)

    def _add_system_message(self, messages: Messages, content: str) -> None:
        self._add_message(messages, self._build_message("system", content))

    def _add_user_message(self, messages: Messages, content: Any) -> None:
        self._add_message(messages, self._build_message("user", content))

    def _add_user_text_message(self, messages: Messages, content: str) -> None:
        self._add_user_message(messages, self._build_text_content(content))

    def _prepare_messages(
        self,
        query: str,
        image: Image.Image,
        history: Messages,
        rag_context: str | None = None,
        resize_image: bool = False,
    ) -> Messages:
        """Prepare messages with standardized order: System → Image → RAG → History →
        Query."""
        messages: Messages = []
        # 1. System prompt handled separately in API call
        system_prompt = SYSTEM_MULTI_PROMPT if history else SYSTEM_PROMPT
        self._add_system_message(messages, self._build_text_content(system_prompt))
        # 2. Image (if any)
        if image is not None:
            if resize_image:
                image = resize_image_if_needed(
                    image, max_size_mb=5.0, max_dimension_px=8000
                )
            self._add_user_message(messages, self._build_image_content(image))
        # 3. RAG context (if any)
        if rag_context:
            self._add_user_text_message(messages, rag_context)
        # 4. History (if any)
        if history:
            if hasattr(self, "_get_prompt_summary"):
                self._add_user_text_message(messages, "<conversation_history>")
            for message in history:
                self._add_message(messages, message)
            if hasattr(self, "_get_prompt_summary"):
                self._add_user_text_message(messages, "</conversation_history>")
        # 5. Critical message for VlmAgent
        if hasattr(self, "_get_prompt_summary"):
            self._add_user_text_message(
                messages, self._get_prompt_summary(rag_context, history)
            )
        # 6. Query
        self._add_user_text_message(messages, query)
        return messages

    @abstractmethod
    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[Messages],
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
        raise NotImplementedError("Subclasses must implement this method")
