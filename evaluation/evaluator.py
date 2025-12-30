"""
Main evaluator class for CRAG-MM dataset.

This module contains the CRAGEvaluator class that orchestrates the evaluation process,
including response generation, semantic evaluation, and results calculation.
"""

import json
import os
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import tqdm
from example_agents.base_agent import BaseAgent
from utils.retriever import get_rag_context
from utils.crag_batch_iterator import CRAGTurnBatchIterator
from cragmm_search.search import UnifiedSearchPipeline
from datasets import Dataset

from evaluation.config import (
    MAX_BATCH_SIZE,
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
from evaluation.llm_judge import LLMJudge
from evaluation.evaluation_utils import calculate_scores
from rich.console import Console
from tokenizers import Tokenizer

console = Console()


class CRAGEvaluator:
    """
    A class to evaluate an agent on the CRAG-MM dataset.

    This evaluator generates responses, evaluates them (optionally using a semantic evaluation model),
    computes multi-turn conversation metrics, and (optionally) saves the results.
    """

    def __init__(
        self,
        dataset: Dataset | None,
        agent: BaseAgent | None,
        eval_model_name: str | None = None,
        eval_api_key: str | None = None,
        num_conversations: int | None = None,
        image_search: bool = False,
        web_search: bool = False,
        search_pipeline: UnifiedSearchPipeline | None = None,
        show_progress: bool = True,
        num_workers: int = 1,
        dataset_type: str = "single-turn",
        full_query_path: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.agent = agent
        self.eval_model_name = eval_model_name
        self.eval_api_key = eval_api_key
        self.num_conversations = num_conversations
        self.image_search = image_search
        self.web_search = web_search
        self.search_pipeline = search_pipeline
        self.show_progress = show_progress
        self.num_workers = num_workers
        self.elasped_time = None
        self.dataset_type = dataset_type
        self.full_query_file = full_query_path

        # Internal state for evaluation; these are set during initialization
        self.batch_iterator: CRAGTurnBatchIterator | None = None
        self.conversations_count: int = 0
        self.agent_response_map: dict[str, str] = {}
        self.all_turn_data: list[dict[str, any]] = []
        self.session_ids_evaluated: set[str] = set()

        self.tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.tokenizer.enable_truncation(max_length=MAX_RESPONSE_LENGTH_IN_TOKENS)

        # Initialize LLM Judge if evaluation model is provided
        self.llm_judge: LLMJudge | None = None
        if self.eval_model_name and self.eval_api_key:
            self.llm_judge = LLMJudge(self.eval_api_key, self.eval_model_name)

    def evaluate_response(self, crag_turn_data: dict[str, any]) -> dict[str, any]:
        """
        Evaluate a single response and return evaluation results.

        Args:
            crag_turn_data: A dictionary containing query, ground truth, and agent response.

        Returns:
            A dictionary with evaluation results added to crag_turn_data.
        """
        agent_response = crag_turn_data["agent_response"]
        ground_truth = crag_turn_data["ground_truth"]
        query = crag_turn_data["query"]

        agent_response_clean = clean_text(agent_response)

        is_idk = (
            "i dont know" in agent_response_clean
            or "i do not know" in agent_response_clean
        )

        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_semantically_correct = False
        api_response = None

        # Begin by assuming exact match correctness
        is_correct = is_exact_match

        # Use semantic evaluation if not an exact match and an evaluation model is provided.
        if not is_idk and not is_exact_match and self.llm_judge:
            api_response = self.llm_judge.evaluate(query, ground_truth, agent_response)
            if api_response:
                is_semantically_correct = api_response["accuracy"]
                is_correct = is_semantically_correct
        if is_exact_match:
            is_semantically_correct = True

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_miss": is_idk,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response if api_response else None,
        }

    def initialize_for_generation(self) -> None:
        """
        Initialize variables needed for generation-only mode.

        Lighter version of initialize_evaluation() that only sets up
        what's needed for response generation.
        """
        if self.dataset is None or self.agent is None:
            raise ValueError("Dataset and agent are required for generation")

        console.print("[blue]Starting generation mode[/blue]")

        self.conversations_count = (
            len(self.dataset)
            if self.num_conversations is None
            else min(self.num_conversations, len(self.dataset))
        )
        batch_size = int(
            np.clip(self.agent.get_batch_size(), MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        )
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        # Instantiate the CRAG turn based batch iterator
        self.batch_iterator = CRAGTurnBatchIterator(
            dataset=self.dataset,
            batch_size=batch_size,
            full_query_file=self.full_query_file,
            shuffle=False,
            sessions_to_skip=SESSIONS_TO_SKIP.get(self.dataset_type, []),
            sessions_to_update=SESSIONS_TO_UPDATE.get(self.dataset_type, {}),
        )

        # Store feature vocabularies for domain mapping
        self.feature_vocabularies = prepare_feature_vocabularies(self.dataset)

    def initialize_evaluation(self) -> None:
        """
        Initialize variables needed for full evaluation (generation + evaluation).

        This method sets internal state including the batch iterator, conversation count,
        agent response map, and turn data list.
        """
        console.print(
            f"[blue]Starting evaluation with {self.num_workers} workers[/blue]"
        )
        if self.eval_model_name:
            console.print(
                f"[blue]Using semantic evaluation with model: {self.eval_model_name}[/blue]"
            )

        # Use the same initialization logic as generation mode
        self.initialize_for_generation()

    def generate_agent_responses(
        self, progress_callback: Callable[[int, int], None] = None
    ) -> None:
        """
        Phase 1: Generate agent responses for each turn in the dataset.

        This method iterates over the dataset batches using the internal batch iterator and updates the evaluator's state
        with agent responses and turn data.
        """
        start_time = time.time()
        if self.batch_iterator is None:
            raise ValueError(
                "Batch iterator is not initialized. Please call initialize_evaluation() first."
            )
        if self.agent is None:
            raise ValueError("Agent is required for generate_agent_responses()")

        for batch_idx, batch in enumerate(
            tqdm.tqdm(
                self.batch_iterator,
                desc="Generating responses",
                disable=not self.show_progress,
            )
        ):
            interaction_ids = batch["interaction_ids"]
            queries = batch["queries"]
            images = batch["images"]
            conversation_histories = batch["conversation_histories"]
            full_queries = batch["full_queries"]

            message_histories = []
            interaction_id_histories = []
            # Build message histories for multi-turn conversations
            for conversation_history in conversation_histories:
                message_history = []
                interaction_id_history = []
                for turn in conversation_history:
                    turn_interaction_id = turn["interaction_id"]
                    turn_agent_response = self.agent_response_map.get(
                        turn_interaction_id
                    )
                    if not turn_agent_response:
                        raise AssertionError(
                            f"Agent response not found for turn {turn_interaction_id}. "
                            "Did you shuffle the multi-turn conversations by mistake?"
                        )
                    message_history.append({"role": "user", "content": turn["query"]})
                    message_history.append(
                        {"role": "assistant", "content": turn_agent_response}
                    )
                    interaction_id_history.append(turn_interaction_id)
                message_histories.append(message_history)
                interaction_id_histories.append(interaction_id_history)

            # Convert domain indices to domain names before passing to agent
            domain_names = convert_domain_indices_to_names(
                batch["domains"], self.feature_vocabularies
            )

            rag_context_list = [
                get_rag_context(
                    query,
                    image,
                    self.search_pipeline,
                    self.image_search,
                    self.web_search,
                    full_query,
                )
                for query, image, full_query in zip(queries, images, full_queries)
            ]

            # Generate responses for the current batch
            agent_responses = self.agent.batch_generate_response(
                queries,
                images,
                message_histories,
                rag_context_list=rag_context_list,
                domains=domain_names,
            )
            agent_responses = self.truncate_agent_responses(
                agent_responses
            )  # Truncase each response to the maximum allowed length (75 tokens)

            # Collect responses and add evaluation data
            for idx, interaction_id in enumerate(interaction_ids):
                agent_response = agent_responses[idx]
                self.agent_response_map[interaction_id] = agent_response
                self.all_turn_data.append(
                    {
                        "session_id": batch["session_ids"][idx],
                        "interaction_id": interaction_id,
                        "turn_idx": batch["turn_idxs"][idx],
                        "is_ego": batch["image_urls"][idx] is None,
                        "image_quality": batch["image_qualities"][idx],
                        "query_category": batch["query_categories"][idx],
                        "domain": batch["domains"][idx],
                        "dynamism": batch["dynamisms"][idx],
                        "query": queries[idx],
                        "ground_truth": batch["answers"][idx],
                        "agent_response": agent_response,
                        "total_turn_count": batch["total_turn_counts"][idx],
                        "interaction_id_history": interaction_id_histories[idx],
                    }
                )
                self.session_ids_evaluated.add(batch["session_ids"][idx])

            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                console.print(
                    f"[yellow]Already evaluated {len(self.session_ids_evaluated)} conversations. Abruptly stopping evaluation.[/yellow]"
                )
                break

        end_time = time.time()
        self.elasped_time = end_time - start_time
        print(f"Elasped time: Function ran for {self.elasped_time:.4f} seconds")

    def evaluate_agent_responses(
        self,
        turn_data: list[dict[str, any]],
        progress_callback: Callable[[int, int], None] = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
        """
        Phase 2: Evaluate agent responses and calculate scores.

        This method uses a thread-based parallel executor to avoid pickling issues.
        Args:
            turn_data: List of turn data including agent responses.
        Returns:
            A tuple containing turn evaluation results and score dictionaries.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.evaluate_response, data) for data in turn_data
            ]
            for future_idx, future in tqdm.tqdm(
                enumerate(as_completed(futures)),
                total=len(futures),
                desc="Evaluating responses",
                disable=not self.show_progress,
            ):
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(future_idx, len(turn_data))

        # Convert the interim evaluation results to a pandas dataframe
        turn_evaluation_results_df = pd.DataFrame(results)
        turn_evaluation_results_df = turn_evaluation_results_df.sort_values(
            by=["session_id", "turn_idx"]
        )

        ego_turn_evaluation_results_df = turn_evaluation_results_df[
            turn_evaluation_results_df["is_ego"] == True
        ]

        all_scores_dictionary = calculate_scores(turn_evaluation_results_df)
        ego_scores_dictionary = calculate_scores(ego_turn_evaluation_results_df)

        turn_evaluation_results = {
            "all": turn_evaluation_results_df,
            "ego": ego_turn_evaluation_results_df,
        }
        score_dictionaries = {
            "all": all_scores_dictionary,
            "ego": ego_scores_dictionary,
        }

        return turn_evaluation_results, score_dictionaries

    def save_results(
        self,
        turn_evaluation_results: dict[str, any],
        scores_dictionary: dict[str, any],
        output_dir: str,
    ) -> None:
        """
        Save evaluation results to the specified directory.

        Args:
            turn_evaluation_results: The evaluation results to save.
            scores_dictionary: The scores dictionary to save.
            output_dir: Path where to save the results.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{output_dir}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        turn_evaluation_results["all"].to_csv(
            os.path.join(output_dir, "turn_evaluation_results_all.csv"), index=False
        )
        turn_evaluation_results["ego"].to_csv(
            os.path.join(output_dir, "turn_evaluation_results_ego.csv"), index=False
        )
        with open(os.path.join(output_dir, "scores_dictionary.json"), "w") as f:
            json.dump(scores_dictionary, f, indent=2)

        if self.elasped_time is not None:
            try:
                time_str = (
                    f"Elasped time: Function ran for {self.elasped_time:.4f} seconds"
                )
                time_file = os.path.join(output_dir, "elasped_time.txt")
                with open(time_file, "w+") as file:
                    file.write(time_str)
            except Exception as e:
                print(f"Elasped time did not write successfully {e}")

    def save_turn_data(self, output_file: str) -> None:
        """
        Save turn data to a JSONL file for later evaluation.

        Args:
            output_file: Path where to save the turn data in JSONL format.
        """
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{output_dir}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        base_file = os.path.basename(output_file)
        output_file = f"{output_dir}/{base_file}"

        with open(output_file, "w") as f:
            for turn_data in self.all_turn_data:
                f.write(json.dumps(turn_data) + "\n")
        console.print(
            f"[green]Saved {len(self.all_turn_data)} turns to {output_file}[/green]"
        )

        if self.elasped_time is not None:
            try:
                time_str = (
                    f"Elasped time: Function ran for {self.elasped_time:.4f} seconds"
                )
                time_file = os.path.join(output_dir, "elasped_time.txt")
                with open(time_file, "w+") as file:
                    file.write(time_str)
            except Exception as e:
                print(f"Elasped time did not write successfully {e}")

    def load_turn_data(self, input_file: str) -> list[dict[str, any]]:
        """
        Load turn data from a JSONL file.

        Args:
            input_file: Path to the JSONL file containing turn data.

        Returns:
            List of turn data dictionaries.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        turn_data = []
        with open(input_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Validate that required fields are present
                    required_fields = [
                        "session_id",
                        "interaction_id",
                        "query",
                        "ground_truth",
                        "agent_response",
                    ]
                    missing_fields = [
                        field for field in required_fields if field not in data
                    ]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    turn_data.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                except ValueError as e:
                    raise ValueError(f"Error on line {line_num}: {e}")

        console.print(f"[green]Loaded {len(turn_data)} turns from {input_file}[/green]")
        return turn_data

    def evaluate_agent(self) -> tuple[dict[str, any], dict[str, any]]:
        """
        Evaluate an agent on a dataset and return performance metrics.

        Returns:
            A tuple containing a dictionary of turn evaluation results and a dictionary of scores.
        """
        # Phase 0: Initialize evaluation state
        self.initialize_evaluation()

        # Phase 1: Generate agent responses (updates internal state)
        def _generation_progress_callback(
            conversations_evaluated: int, total_conversations: int
        ) -> None:
            # Can be useful to track progress of the evaluation
            pass

        self.generate_agent_responses(_generation_progress_callback)

        # Phase 2: Evaluate responses using stored turn data

        def _evaluation_progress_callback(
            turn_evaluated: int, total_turns: int
        ) -> None:
            # Can be useful to track progress of the evaluation
            pass

        turn_evaluation_results, score_dictionaries = self.evaluate_agent_responses(
            self.all_turn_data, _evaluation_progress_callback
        )
        return turn_evaluation_results, score_dictionaries

    def truncate_agent_responses(self, agent_responses: list[str]) -> list[str]:
        """
        Truncate each agent response to the maximum allowed length.
        """
        encodings = self.tokenizer.encode_batch(agent_responses)
        trimmed_agent_responses = [self.tokenizer.decode(enc.ids) for enc in encodings]
        return trimmed_agent_responses
