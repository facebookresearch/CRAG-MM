# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np


def calculate_scores(turn_evaluation_results_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate scores for both single-turn and multi-turn conversations.

    Args:
        turn_evaluation_results_df: DataFrame with evaluation results for turns.
    Returns:
        Dictionary of calculated metrics.
    """
    multi_turn_conversation_score_map: dict[str, float] = {}

    def _set_is_correct_false_after_consecutive(
        group: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Mark as is_miss after consecutive incorrect responses
        and calculate multi-turn conversation score for each conversation.
        """
        group_copy = group.copy().reset_index(drop=True)
        for i in range(1, len(group_copy)):
            if (
                not group_copy.loc[i - 1, "is_correct"]
                and not group_copy.loc[i, "is_correct"]
            ):
                group_copy.loc[i + 1 :, "is_correct"] = False
                group_copy.loc[i + 1 :, "is_exact_match"] = False
                group_copy.loc[i + 1 :, "is_miss"] = True
                group_copy.loc[i + 1 :, "is_semantically_correct"] = False
                break

        group_copy["is_hallucination"] = (
            ~group_copy["is_correct"] & ~group_copy["is_miss"]
        )
        multi_turn_conversation_score = (
            group_copy["is_correct"].mean() - group_copy["is_hallucination"].mean()
        )
        group_copy["multi_turn_conversation_score"] = multi_turn_conversation_score
        session_id = group_copy.iloc[0]["session_id"]
        multi_turn_conversation_score_map[session_id] = multi_turn_conversation_score
        return group_copy

    total = len(turn_evaluation_results_df)
    correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
    correct = turn_evaluation_results_df["is_correct"].sum()
    miss = turn_evaluation_results_df["is_miss"].sum()
    hallucination = total - (correct + miss)

    turn_evaluation_results_df = turn_evaluation_results_df.groupby(
        "session_id", group_keys=False
    )[turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

    exact_match = correct_exact / total
    accuracy = correct / total
    missing = miss / total
    hallucination_rate = hallucination / total
    truthfulness_score = ((2 * correct + miss) / total) - 1 if total > 1 else 0.0
    mean_multi_turn_conversation_score = np.mean(
        list(multi_turn_conversation_score_map.values())
    )

    scores_dictionary = {
        "total": float(total),
        "correct_exact": float(correct_exact),
        "correct": float(correct),
        "miss": float(miss),
        "hallucination": float(hallucination),
        "exact_match": float(exact_match),
        "accuracy": float(accuracy),
        "missing": float(missing),
        "hallucination_rate": float(hallucination_rate),
        "truthfulness_score": float(truthfulness_score),
        "mean_multi_turn_conversation_score": float(mean_multi_turn_conversation_score),
    }

    return scores_dictionary

