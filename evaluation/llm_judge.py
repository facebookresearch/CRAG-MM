# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM-as-a-Judge implementation for semantic evaluation.

This module provides functionality for using OpenAI API to semantically evaluate
agent responses against ground truth answers.
"""

import time
from typing import Tuple

from evaluation.config import MAX_API_RETRIES, MAX_EVAL_TOKENS
from evaluation.evaluation_prompt import CRAGMultiModalPrompts

from openai import OpenAI
from rich.console import Console

console = Console()


class LLMJudge:
    """LLM-as-a-Judge for semantic evaluation of agent responses."""

    def __init__(self, api_key: str, model_name: str):
        """Initialize the LLM Judge.

        Args:
            api_key: OpenAI API key for authentication.
            model_name: Name of the model to use for evaluation (e.g., "gpt-4o").
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.prompt = CRAGMultiModalPrompts()

    def get_system_message(self) -> str:
        """Get the system message for the evaluator.

        Returns:
            System message combining instructions and in-context examples.
        """
        message_system = (
            self.prompt.get_instructions()
            + "\n\n"
            + self.prompt.get_in_context_examples()
        )
        return message_system

    def parse_response(self, eval_result: str) -> Tuple[bool | None, str]:
        """Parse the eval result and return (is_correct, reason).

        Args:
            eval_result: The evaluation result from the LLM.

        Returns:
            Tuple of (is_correct, reason) where is_correct is True/False/None
            and reason is the full evaluation result.
        """
        try:
            if "Result: WRONG" in eval_result or "WRONG" in eval_result.split("\n")[-1]:
                return (False, eval_result)
            elif (
                "Result: CORRECT" in eval_result
                or "CORRECT" in eval_result.split("\n")[-1]
            ):
                return (True, eval_result)
            else:
                return (None, eval_result)
        except Exception as e:
            return (None, eval_result)

    def attempt_api_call(
        self,
        messages: list,
        max_retries: int = MAX_API_RETRIES,
    ) -> dict[str, any] | None:
        """Attempt an API call to the OpenAI API with retries.

        Args:
            messages: List of message objects for the conversation.
            max_retries: Maximum number of retry attempts before giving up.

        Returns:
            Dictionary with evaluation result if successful, None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=MAX_EVAL_TOKENS,
                )
                eval_result = completion.choices[0].message.content
                is_correct, reason = self.parse_response(eval_result)
                return {"accuracy": is_correct, "eval_result": reason}
            except Exception as e:
                error_message = (
                    f"API call failed on attempt {attempt + 1}/{max_retries}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    console.print(
                        f"[red]Failed after {MAX_API_RETRIES} attempts: {str(e)}[/red]"
                    )
                else:
                    console.print(f"[yellow]{error_message}, retrying...[/yellow]")
                    time.sleep(2**attempt)

        return None

    def evaluate(
        self, query: str, ground_truth: str, agent_response: str
    ) -> dict[str, any] | None:
        """Evaluate an agent response against ground truth.

        Args:
            query: The query/question asked.
            ground_truth: The expected correct answer.
            agent_response: The agent's response to evaluate.

        Returns:
            Dictionary with evaluation result or None if evaluation fails.
        """
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {
                "role": "user",
                "content": f"Your task is to judge if the prediction is correct or not based on the ground truth answer. Now your turn:\n\nQuestion: {query}\nGround Truth: {ground_truth}\nPrediction: {agent_response}\nGive your output below:\n",
            },
        ]
        return self.attempt_api_call(messages)
