# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluator Script for CRAG-MM dataset

This script evaluates an agent (using a user-provided agent `UserAgent` as configured in `agents/user_config.py`)
on the CRAG-MM dataset. It generates responses, evaluates them (using an optional semantic evaluation model via OpenAI API),
computes multi-turn conversation metrics, and optionally saves the results.

This is a lightweight CLI wrapper that imports the evaluation logic from the evaluation module.
"""

import argparse
import os

# Load environment variables first
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from utils.utils import display_results, ensure_crag_cache_dir_is_configured

load_dotenv()

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "true"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


from example_agents.user_config import get_agent_from_model_name
from cragmm_search.search import UnifiedSearchPipeline
from datasets import load_dataset
from evaluation import CRAGEvaluator, DEFAULT_EVAL_MODEL, DEFAULT_NUM_WORKERS

CACHE_DIR = ensure_crag_cache_dir_is_configured()
console = Console()


def setup_eval_model(eval_model: str) -> str | None:
    """Helper function to setup evaluation model with warning if disabled."""
    if eval_model.lower() == "none":
        console.print(
            Panel(
                "[bold red]WARNING: SEMANTIC EVALUATION IS DISABLED[/bold red]\n\n"
                "No calls to LLM-as-a-Judge will be made!\n"
                "Results will rely only on exact string matching.",
                title="[bold red]ATTENTION[/bold red]",
                border_style="red",
                width=100,
                padding=(2, 5),
                expand=False,
            )
        )
        return None
    return eval_model


def display_and_save_results(
    evaluator: CRAGEvaluator,
    turn_evaluation_results: dict[str, pd.DataFrame],
    score_dictionaries: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    """Helper function to display and optionally save results."""
    # Display all results
    display_results(
        console,
        turn_evaluation_results["all"],
        score_dictionaries["all"],
        display_conversations=args.display_conversations,
        is_ego=False,
        is_multi_turn=getattr(args, "dataset_type", "multi-turn") == "multi-turn",
    )

    # Display ego results if available
    if len(turn_evaluation_results["ego"]) > 0:
        display_results(
            console,
            turn_evaluation_results["ego"],
            score_dictionaries["ego"],
            display_conversations=args.display_conversations,
            is_ego=True,
            is_multi_turn=getattr(args, "dataset_type", "multi-turn") == "multi-turn",
        )

    # Save results if output directory specified
    if args.output_dir:
        evaluator.save_results(
            turn_evaluation_results, score_dictionaries, args.output_dir
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on the CRAG-MM dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="single-turn",
        choices=["single-turn", "multi-turn"],
        help="Dataset type to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use ('validation', 'public_test')",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=-1,
        help=(
            "Number of conversations to evaluate (default: -1). -1 evaluates all"
            " conversations, while a positive number evaluates that many conversations."
        ),
    )
    parser.add_argument(
        "--suppress-web-search-api",
        action="store_true",
        help="Suppress web search API when calling the agent",
    )
    parser.add_argument(
        "--suppress-image-search-api",
        action="store_true",
        help="Suppress image search API when calling the agent",
    )
    parser.add_argument(
        "--display-conversations",
        type=int,
        default=10,
        help="Number of evaluation examples to show",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help=(
            "OpenAI model for semantic evaluation. Pass 'None' to disable semantic"
            " evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Path to save turn evaluation results, scores dictionary, and generated responses (response.jsonl)",
    )
    parser.add_argument(
        "--no_progress", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v0.1.2",
        help="Dataset revision/version to use when loading from HuggingFace",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of worker processes for parallel evaluation (default:"
            f" {DEFAULT_NUM_WORKERS})"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name to feed into vllm. The agent type is automatically inferred from the model name (e.g., 'llama', 'claude', 'gpt', 'gemini', 'pixtral', 'internvl', 'qwen').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["generate", "evaluate", "both"],
        help=(
            "Mode to run: 'generate' (generation only), 'evaluate' (evaluation only),"
            " or 'both' (default: both)"
        ),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Path to load turn data from (JSONL format). Required for --mode evaluate."
        ),
    )
    parser.add_argument(
        "--full-query-path",
        type=str,
        default=None,
        help="Path to the saved full-query path, required to update web search query.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define the response file path
    response_file = os.path.join(args.output_dir, "response.jsonl")

    # Validate mode-specific arguments
    if args.mode == "generate" and os.path.exists(response_file):
        raise ValueError(
            f"Error: {response_file} already exists and will be overwritten in generate mode. "
            "Please remove it or use a different output directory."
        )
    if args.mode == "evaluate" and not args.input_file:
        console.print(
            "[red]Error: --input-file is required when using --mode evaluate[/red]"
        )
        return

    if args.mode != "generate":
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")

    # For evaluate-only mode, we don't need to load dataset or initialize agent
    if args.mode == "evaluate":
        console.print("[bold blue]Running evaluation-only mode...[/bold blue]")

        # Setup evaluation model with warning if needed
        args.eval_model = setup_eval_model(args.eval_model)

        # Create minimal evaluator for evaluation-only
        evaluator = CRAGEvaluator(
            dataset=None,
            agent=None,
            eval_model_name=args.eval_model,
            eval_api_key=OPENAI_API_KEY,
            num_conversations=None,
            show_progress=not args.no_progress,
            num_workers=args.num_workers,
            dataset_type=args.dataset_type,
        )

        # Load turn data and evaluate
        turn_data = evaluator.load_turn_data(args.input_file)
        turn_evaluation_results, score_dictionaries = (
            evaluator.evaluate_agent_responses(turn_data)
        )

        # Display and save results using helper function
        display_and_save_results(
            evaluator, turn_evaluation_results, score_dictionaries, args
        )
        return

    # For generate or both modes, we need dataset and agent
    console.print(f"[bold blue]Loading {args.dataset_type} dataset...[/bold blue]")
    repo_name = f"crag-mm-2025/crag-mm-{args.dataset_type}-public"
    console.print(
        f"[bold green]Loading from HuggingFace:[/bold green] {repo_name} (revision:"
        f" {args.revision})"
    )
    dataset = load_dataset(repo_name, revision=args.revision, cache_dir=CACHE_DIR)
    available_splits = list(dataset.keys())
    split_to_use = args.split if args.split in available_splits else available_splits[0]
    console.print(
        f"[bold green]Using split:[/bold green] '{split_to_use}' with"
        f" {len(dataset[split_to_use])} examples"
    )

    # Setup evaluation model with warning if needed
    args.eval_model = setup_eval_model(args.eval_model)

    if args.num_conversations == -1:
        args.num_conversations = len(dataset[split_to_use])

    # Suppress web search API if the flag is set - useful for Task 1 (Single-source
    # Augmentation)
    search_api_text_model_name = "BAAI/bge-large-en-v1.5"
    search_api_image_model_name = "openai/clip-vit-large-patch14-336"
    search_api_web_hf_dataset_id = "crag-mm-2025/web-search-index-public-test"
    search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-public-test"

    if args.suppress_web_search_api:
        # Suppress web search API - useful for Task 1 (Single-source Augmentation)
        search_api_web_hf_dataset_id = None

    search_pipeline = UnifiedSearchPipeline(
        text_model_name=search_api_text_model_name,
        image_model_name=search_api_image_model_name,
        web_hf_dataset_id=search_api_web_hf_dataset_id,
        image_hf_dataset_id=search_api_image_hf_dataset_id,
    )

    UserAgent = get_agent_from_model_name(args.model_name)

    evaluator = CRAGEvaluator(
        dataset=dataset[split_to_use],
        agent=UserAgent(
            model_name=args.model_name, is_multi_turn=args.dataset_type == "multi-turn"
        ),
        image_search=not args.suppress_image_search_api,
        web_search=not args.suppress_web_search_api,
        search_pipeline=search_pipeline,
        eval_model_name=args.eval_model,
        eval_api_key=OPENAI_API_KEY,
        num_conversations=args.num_conversations,
        show_progress=not args.no_progress,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        full_query_path=args.full_query_path,
    )

    if args.mode == "generate":
        # Generation-only mode
        console.print("[bold blue]Running generation-only mode...[/bold blue]")
        evaluator.initialize_for_generation()
        evaluator.generate_agent_responses()
        evaluator.save_turn_data(response_file)
        console.print(
            f"[green]Generation complete. Results saved to {response_file}[/green]"
        )
        return

    # Both mode (default behavior)
    console.print("[bold blue]Running both generation and evaluation...[/bold blue]")
    turn_evaluation_results, score_dictionaries = evaluator.evaluate_agent()

    # Save intermediate turn data
    evaluator.save_turn_data(response_file)

    # Display and save results using helper function
    display_and_save_results(
        evaluator, turn_evaluation_results, score_dictionaries, args
    )


if __name__ == "__main__":
    main()
