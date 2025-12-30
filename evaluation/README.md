# Evaluation Module

This folder contains the evaluation framework for the CRAG-MM (Comprehensive RAG Benchmark - MultiModal) dataset. The evaluator implements an **LLM-as-a-Judge** approach to automatically assess whether RAG agents successfully answer queries when compared to expected ground truth answers.

## Overview

The evaluation pipeline supports:
- **Exact match evaluation**: Direct string comparison between agent responses and ground truth
- **Semantic evaluation**: LLM-based judgment for responses that are correct but phrased differently
- **Multi-turn conversation scoring**: Metrics that account for conversation flow and consecutive errors
- **Parallel evaluation**: Thread-based concurrent processing for faster evaluation

## Module Structure

```
evaluation/
├── config.py           # Configuration constants and parameters
├── dataset_utils.py    # Dataset preprocessing utilities
├── evaluator.py        # Main CRAGEvaluator orchestration class
├── llm_judge.py        # LLM-as-a-Judge implementation
└── README.md           # This file
```

### `config.py`

Centralizes all configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_EVAL_MODEL` | `"gpt-4o"` | Default model for semantic evaluation |
| `MAX_API_RETRIES` | `3` | Maximum retry attempts for API calls |
| `DEFAULT_NUM_WORKERS` | `1` | Default number of parallel workers |
| `MIN_BATCH_SIZE` | `1` | Minimum batch size for processing |
| `MAX_BATCH_SIZE` | `16` | Maximum batch size (configurable via `MAX_BATCH_SIZE` env var) |
| `MAX_RESPONSE_LENGTH_IN_TOKENS` | `75` | Maximum tokens for agent responses |
| `MAX_EVAL_TOKENS` | `1024` | Maximum tokens for evaluation responses |

### `dataset_utils.py`

Provides utilities for dataset preparation:
- `prepare_feature_vocabularies()`: Extracts feature vocabularies for category encoding
- `convert_domain_indices_to_names()`: Maps domain indices to human-readable names
- `clean_text()`: Normalizes text for comparison (lowercase, remove special characters)

### `evaluator.py`

The main `CRAGEvaluator` class that orchestrates the evaluation process:
- Generates agent responses for dataset queries
- Evaluates responses using exact match and/or semantic evaluation
- Calculates comprehensive metrics
- Saves results in structured formats

### `llm_judge.py`

Implements the `LLMJudge` class for semantic evaluation:
- Uses OpenAI API to compare agent responses against ground truth
- Parses evaluation results into CORRECT/WRONG verdicts
- Handles API retries with exponential backoff

## Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRAGEvaluator.evaluate_agent()              │
├─────────────────────────────────────────────────────────────────┤
│  Phase 0: initialize_evaluation()                               │
│    └── Set up batch iterator, feature vocabularies              │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: generate_agent_responses()                            │
│    └── Generate responses for each turn in dataset              │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: evaluate_agent_responses()                            │
│    ├── Exact match check                                        │
│    ├── Semantic evaluation via LLMJudge (if not exact match)    │
│    └── Calculate scores                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Output Format

### Turn Evaluation Results (CSV)

The evaluator outputs two CSV files:
- `turn_evaluation_results_all.csv`: Results for all turns
- `turn_evaluation_results_ego.csv`: Results for ego-centric turns only

Each CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | string | Unique identifier for the conversation session |
| `interaction_id` | string | Unique identifier for the specific turn |
| `turn_idx` | int | Turn index within the conversation (0-indexed) |
| `is_ego` | bool | Whether this is an ego-centric query (no external image) |
| `image_quality` | int | Quality rating of the associated image |
| `query_category` | int | Category index of the query type |
| `domain` | int | Domain index (e.g., automotive, sports) |
| `dynamism` | int | Dynamism level of the content |
| `query` | string | The original query/question |
| `ground_truth` | string | Expected correct answer |
| `agent_response` | string | The agent's generated response |
| `total_turn_count` | int | Total number of turns in the conversation |
| `interaction_id_history` | list | List of previous interaction IDs in the conversation |
| `is_exact_match` | bool | Whether response exactly matches ground truth |
| `is_correct` | bool | Whether response is considered correct (exact or semantic) |
| `is_miss` | bool | Whether agent responded with "I don't know" |
| `is_semantically_correct` | bool | Whether response is semantically correct |
| `api_response` | dict/null | Raw LLM judge response (if semantic evaluation was used) |

### Scores Dictionary (JSON)

The `scores_dictionary.json` file contains aggregated metrics:

```json
{
  "all": {
    "total": 1000.0,
    "correct_exact": 450.0,
    "correct": 720.0,
    "miss": 80.0,
    "hallucination": 200.0,
    "exact_match": 0.45,
    "accuracy": 0.72,
    "missing": 0.08,
    "hallucination_rate": 0.20,
    "truthfulness_score": 0.52,
    "mean_multi_turn_conversation_score": 0.65
  },
  "ego": {
    // Same structure for ego-centric turns only
  }
}
```

#### Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| `total` | N | Total number of evaluated turns |
| `correct_exact` | Σ(is_exact_match) | Count of exact string matches |
| `correct` | Σ(is_correct) | Count of correct responses (exact + semantic) |
| `miss` | Σ(is_miss) | Count of "I don't know" responses |
| `hallucination` | total - correct - miss | Count of incorrect confident responses |
| `exact_match` | correct_exact / total | Exact match rate |
| `accuracy` | correct / total | Overall accuracy rate |
| `missing` | miss / total | Missing/abstention rate |
| `hallucination_rate` | hallucination / total | Hallucination rate |
| `truthfulness_score` | (2 × correct + miss) / total - 1 | Score balancing correctness and abstention |
| `mean_multi_turn_conversation_score` | Mean of per-conversation scores | Average multi-turn performance |

### Turn Data (JSONL)

For generation-only mode, turn data is saved in JSONL format with one JSON object per line:

```json
{
  "session_id": "abc123",
  "interaction_id": "xyz789",
  "turn_idx": 0,
  "is_ego": true,
  "image_quality": 2,
  "query_category": 1,
  "domain": 3,
  "dynamism": 0,
  "query": "What color is this car?",
  "ground_truth": "red",
  "agent_response": "The car appears to be red.",
  "total_turn_count": 3,
  "interaction_id_history": []
}
```

## Usage

### Basic Evaluation

```python
from evaluation.evaluator import CRAGEvaluator
from datasets import load_dataset

# Load dataset and agent
dataset = load_dataset("your-dataset")
agent = YourAgent()

# Create evaluator with semantic evaluation
evaluator = CRAGEvaluator(
    dataset=dataset,
    agent=agent,
    eval_model_name="gpt-4o",
    eval_api_key="your-api-key",
    num_conversations=100,
    num_workers=4,
)

# Run evaluation
turn_results, scores = evaluator.evaluate_agent()

# Save results
evaluator.save_results(turn_results, scores, output_dir="./results")
```

### Generation-Only Mode

```python
# For generating responses without evaluation
evaluator = CRAGEvaluator(
    dataset=dataset,
    agent=agent,
    num_conversations=100,
)

evaluator.initialize_for_generation()
evaluator.generate_agent_responses()
evaluator.save_turn_data("./output/turn_data.jsonl")
```

### Evaluation from Saved Turn Data

```python
# Evaluate previously generated responses
evaluator = CRAGEvaluator(
    dataset=None,
    agent=None,
    eval_model_name="gpt-4o",
    eval_api_key="your-api-key",
    num_workers=8,
)

turn_data = evaluator.load_turn_data("./output/turn_data.jsonl")
turn_results, scores = evaluator.evaluate_agent_responses(turn_data)
```

## Multi-Turn Conversation Scoring

For multi-turn conversations, the evaluator applies special rules:
- After **two consecutive incorrect responses**, all subsequent turns are marked as missed
- This reflects that conversation quality degrades after repeated failures
- The `multi_turn_conversation_score` is calculated as: `mean(is_correct) - mean(is_hallucination)` per conversation

## LLM Judge Details

The semantic evaluator uses the following judgment logic:

1. If response contains "I don't know" variants → marked as `is_miss`
2. If response exactly matches ground truth → marked as correct (no API call needed)
3. Otherwise → LLM judge compares response against ground truth

The LLM judge returns:
- `CORRECT`: Response conveys the same meaning as ground truth
- `WRONG`: Response contradicts or differs from ground truth

## Configuration via Environment Variables

| Variable | Description |
|----------|-------------|
| `MAX_BATCH_SIZE` | Override default maximum batch size (default: 16) |
