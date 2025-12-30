from .closed_model_agents import ClaudeAgent, GeminiAgent, GPTAgent
from .open_source_model_agents import InternVLAgent, LlamaAgent, PixtralAgent, QwenAgent

# Mapping from model name keywords to agent names
MODEL_KEYWORD_TO_AGENT = {
    "pixtral": PixtralAgent,
    "claude": ClaudeAgent,
    "gpt": GPTAgent,
    "gemini": GeminiAgent,
    "llama": LlamaAgent,
    "internvl": InternVLAgent,
    "qwen": QwenAgent,
}


def get_agent_from_model_name(model_name: str):
    """
    Parse the model name to extract the corresponding agent.

    Args:
        model_name: The model name string (e.g., "meta-llama/Llama-3.2-11B-Vision-Instruct")

    Returns:
        The agent class corresponding to the model name.

    Raises:
        ValueError: If no matching agent is found for the given model name.
    """
    model_name_lower = model_name.lower()

    for keyword, agent in MODEL_KEYWORD_TO_AGENT.items():
        if keyword in model_name_lower:
            return agent

    supported_keywords = ", ".join(MODEL_KEYWORD_TO_AGENT.keys())
    raise ValueError(
        f"Cannot find a corresponding agent for model '{model_name}'. "
        f"Model name must contain one of the following keywords: {supported_keywords}"
    )
