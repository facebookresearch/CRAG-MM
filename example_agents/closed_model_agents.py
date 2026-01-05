# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from abc import abstractmethod
from concurrent.futures import as_completed, ThreadPoolExecutor
from itertools import zip_longest
from typing import Any

import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure
from PIL import Image
from vllm.multimodal.utils import encode_image_base64

from .base_agent import BaseAgent, Message, Messages
from .prompt_constants import SYSTEM_MULTI_PROMPT, SYSTEM_PROMPT


class ClosedModelBaseAgent(BaseAgent):
    """
    Base class for closed model agents (Claude, Gemini, GPT) with common functionality.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        api_key = os.environ.get(self.api_key_env_var, None)
        if api_key is None:
            raise ValueError(f"{self.api_key_env_var} environment variable is not set!")
        self._setup_client(api_key)

    @property
    @abstractmethod
    def api_key_env_var(self) -> str:
        """Get the environment variable name for the API key."""
        pass

    @abstractmethod
    def _setup_client(self, api_key: str):
        """Setup the API client for the specific model provider."""
        pass

    def _build_image_content(self, image: Image.Image) -> list[dict[str, Any]]:
        """Generate image content in the format expected by the model provider."""
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_base64(image)}"
                },
            }
        ]

    @abstractmethod
    def _make_api_call(self, messages: Any) -> str:
        """Make API call to the model provider."""
        pass

    def get_batch_size(self) -> int:
        return 16

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[Messages],
        rag_context_list: list[str],
        **kwargs,
    ) -> list[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        """
        responses = [None] * len(queries)  # Pre-allocate response list

        def process_single_query(idx, query, image, history, rag_context):
            """Process a single query with RAG context."""
            try:
                messages = self._prepare_messages(query, image, history, rag_context)

                # Make API call with optimized retry logic and exponential backoff.
                last_exception = None
                # Exponential backoff delays: 0.1s, 0.5s
                retry_delays = [0.1, 0.5]
                for attempt in range(3):
                    try:
                        response = self._make_api_call(messages)
                        return idx, response
                    except Exception as e:
                        error_str = str(e)
                        # Check if this is an image size error
                        if (
                            isinstance(self, ClaudeAgent)
                            and "image exceeds" in error_str.lower()
                        ):
                            print(f"Claude API image size error: {error_str}")
                            # If we have the original query parameters, retry with
                            # resized image
                            print("Retrying with resized image...")
                            messages = self._prepare_messages(
                                query,
                                image,
                                history,
                                rag_context,
                                resize_image=True,
                            )
                            try:
                                response = self._make_api_call(messages)
                            except Exception as resize_e:
                                print(
                                    "Claude API error even after resizing: "
                                    f"{str(resize_e)}"
                                )
                                last_exception = resize_e
                                break
                        else:
                            print(f"API error (attempt {attempt + 1}/3): {e}")
                            last_exception = e
                            if attempt < 2:  # Don't sleep if the last attempt fails
                                time.sleep(retry_delays[attempt])
                return (
                    idx,
                    f"[API error after 3 attempts: {str(last_exception)[:100]}...]",
                )
            except Exception as e:
                print(f"Error processing query {idx}: {e}")
                return idx, f"[Error processing query: {e}]"

        # Use optimized ThreadPoolExecutor for concurrent API calls
        # Use deterministic concurrency based on batch size (25% of batch size, min 1,
        # max 8)
        batch_size = self.get_batch_size()
        max_workers = max(
            1,
            min(
                batch_size // 4,  # 25% of batch size
                8,  # Cap at 8 to avoid overwhelming APIs
                len(queries),  # Don't create more workers than queries
            ),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_query, idx, query, image, history, rag_context
                ): idx
                for idx, (query, image, history, rag_context) in enumerate(
                    zip_longest(queries, images, message_histories, rag_context_list)
                )
            }

            # Collect results as they complete with progress tracking
            for future in as_completed(future_to_idx):
                try:
                    idx, response = future.result(
                        timeout=120
                    )  # 2 minute timeout per query
                    responses[idx] = response
                except Exception as e:
                    idx = future_to_idx[future]
                    responses[idx] = f"[Timeout or error: {e}]"

        return responses


class ClaudeAgent(ClosedModelBaseAgent):
    """
    Agent for Anthropic Claude models (Claude 3 family, etc).
    """

    def __init__(self, model_name: str = "claude-3-7-sonnet-latest", **kwargs):
        super().__init__(model_name, **kwargs)

    @property
    def api_key_env_var(self) -> str:
        """Environment variable name for the Anthropic API key."""
        return "ANTHROPIC_API_KEY"

    def _setup_client(self, api_key: str):
        """Setup Claude API client."""
        self.client = Anthropic(api_key=api_key)

    def _build_image_content(self, image: Image.Image) -> list[dict[str, Any]]:
        """Generate image content in the format expected by the model provider."""
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image_base64(image),
                },
            }
        ]

    def _make_api_call(self, messages: list[dict]) -> str:
        """Make Claude API call."""
        system_prompt = messages.pop(0)["content"]  # Extract system prompt
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
        )
        content = response.content
        if isinstance(content, list):
            return " ".join([c.text for c in content if c.type == "text"])
        else:
            return str(content)


class GeminiAgent(ClosedModelBaseAgent):
    """
    Agent for Google Gemini models (Gemini Pro, Gemini 2.5 Flash, etc).
    """

    def __init__(
        self, model_name: str = "gemini-2.5-flash", is_multi_turn: bool = False
    ):
        self.is_multi_turn = is_multi_turn
        super().__init__(model_name)

    @property
    def api_key_env_var(self) -> str:
        """Environment variable name for the Google API key."""
        return "GOOGLE_API_KEY"

    def _setup_client(self, api_key: str):
        """Setup Gemini API client."""
        configure(api_key=api_key)
        self.model = GenerativeModel(
            self.model_name,
            system_instruction=(
                f"{SYSTEM_MULTI_PROMPT if self.is_multi_turn else SYSTEM_PROMPT}"
                "\n\nIf the content appears to be from copyrighted sources (like "
                "Wikipedia), explain or summarize the idea instead of quoting "
                "directly."
            ),
        )

    def _build_text_content(self, text: str) -> dict[str, str]:
        return {"text": text}

    def _build_image_content(self, image: Image.Image) -> dict[str, dict[str, str]]:
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encode_image_base64(image),
            }
        }

    def _add_system_message(self, messages: Messages, content: str) -> None:
        pass

    def _add_message(self, messages: Messages, message: Message) -> None:
        role = "model" if message["role"] == "assistant" else "user"
        if messages and messages[-1]["role"] == role:
            # Append to existing content block
            messages[-1]["parts"].append(message["content"])
        else:
            # Create new content block
            messages.append({"role": role, "parts": [message["content"]]})

    def _make_api_call(self, messages: Messages) -> str:
        """Make Gemini API call with robust finish_reason handling."""
        response = self.model.generate_content(
            contents=messages,
            generation_config={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
            },
        )
        # Robustly check finish_reason for first candidate
        candidate = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason == 4:
                return "[Gemini API blocked response due to copyrighted material.]"
        # If no valid Part/text, return fallback
        if not hasattr(response, "text") or not response.text:
            return "[Gemini API returned no valid text part.]"
        return response.text


class GPTAgent(ClosedModelBaseAgent):
    """
    Agent for OpenAI GPT models (GPT-4o, GPT-5, GPT-5-thinking, GPT-5-mini).
    """

    def __init__(self, model_name: str = "gpt-5-mini", **kwargs):
        super().__init__(model_name, **kwargs)

    @property
    def api_key_env_var(self) -> str:
        """Environment variable name for the OpenAI API key."""
        return "OPENAI_API_KEY"

    def _setup_client(self, api_key: str):
        """Setup OpenAI API client."""
        self.client = openai.OpenAI(api_key=api_key)

    def _make_api_call(self, messages: list[dict]) -> str:
        """Make GPT API call."""
        if self.model_name.startswith("gpt-5"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                top_p=1.0,
                n=1,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                n=1,
            )
        return response.choices[0].message.content


class LlamaClosedAgent(ClosedModelBaseAgent):
    """
    Agent for Llama models through wearables api.
    """

    def __init__(self, model_name: str = "llama3.2-90b-instruct", **kwargs):
        super().__init__(model_name, **kwargs)

    @property
    def api_key_env_var(self) -> str:
        """Environment variable name for the Wearables API key."""
        return "WEARABLES_API_KEY"

    def _setup_client(self, api_key: str):
        """Setup OpenAI API client."""
        self.client = openai.OpenAI(
            api_key=api_key, base_url="https://api.wearables-ape.io/models/v1"
        )

    def _make_api_call(self, messages: list[dict]) -> str:
        """Make GPT API call."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            n=1,
        )
        return response.choices[0].message.content

    def get_batch_size(self) -> int:
        return 1
