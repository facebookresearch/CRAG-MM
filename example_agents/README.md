# RAG Agents

This directory contains Retrieval-Augmented Generation (RAG) agent implementations for the CRAG-MM benchmark. These agents combine Large Language Models (LLMs) with retrieval systems to generate enhanced responses based on user queries, images, and conversation history.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Important Note](#important-note)
- [Guide to Writing Your Own Agents](#guide-to-writing-your-own-agents)
- [Available Agents](#available-agents)
  - [Open-Source Agents](#open-source-agents)
  - [Closed-Source Agents](#closed-source-agents)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Using the LlamaAgent](#using-the-llamaagent)
  - [Building Custom Agents](#building-custom-agents)
  - [Registering Custom Agents](#registering-custom-agents)
- [API Reference](#api-reference)

---

## Directory Structure

```
agents/
├── README.md                    # This documentation file
├── base_agent.py                # Abstract base class for all agents
├── closed_model_agents.py       # API-based agents (Claude, Gemini, GPT)
├── open_source_model_agents.py  # Local model agents (Llama, Pixtral, Qwen, InternVL)
├── random_agent.py              # Reference implementation for testing
└── user_config.py               # Agent registry and model-to-agent mapping
```

### File Descriptions

| File | Description |
|------|-------------|
| `base_agent.py` | Defines `BaseAgent`, the abstract base class that all agents must inherit from. Contains core methods for message preparation and response generation. |
| `closed_model_agents.py` | Implements agents that use external APIs (Anthropic Claude, Google Gemini, OpenAI GPT). These require API keys and make HTTP requests to model providers. |
| `open_source_model_agents.py` | Implements agents that run locally using vLLM with downloaded model weights (Llama, Pixtral, Qwen, InternVL). These require GPU resources. |
| `random_agent.py` | A simple reference implementation that returns random strings. Useful for testing the evaluation pipeline and understanding the agent interface. |
| `user_config.py` | Contains the `MODEL_KEYWORD_TO_AGENT` mapping and `get_agent_from_model_name()` function for dynamic agent selection. |

---

## Important Note

The agents provided in this directory are intended to serve as **baseline and example implementations only**. Users are expected to implement their own agents with:

- **Custom baseline LLM models**: Choose the foundation model that best fits your use case, whether open-source (Llama, Mistral, Qwen) or closed-source (GPT, Claude, Gemini).
- **Different RAG context integration strategies**: Implement your own retrieval and context injection approaches tailored to your specific domain and requirements.

The provided agents demonstrate the expected interface and common patterns, but production deployments should customize the implementation based on specific needs such as:
- Fine-tuned models for domain-specific tasks
- Custom prompting strategies
- Alternative retrieval mechanisms
- Specialized post-processing pipelines

---

## Guide to Writing Your Own Agents

This section provides a brief guide to help you create your own custom agents using the existing framework and utilities.

### Overview

The agent framework follows a simple pattern:
1. **Inherit from `BaseAgent`**: Your agent must extend the `BaseAgent` class from `base_agent.py`
2. **Implement required methods**: Override `get_batch_size()` and `batch_generate_response()`
3. **Register your agent**: Add your agent to `user_config.py` for discovery

### Step-by-Step Guide

#### 1. Create Your Agent File

Create a new Python file in the `agents/` directory (e.g., `my_agent.py`):

```python
from typing import Any
from PIL import Image
from agents.base_agent import BaseAgent, Messages


class MyAgent(BaseAgent):
    def __init__(self, model_name: str = "my-model", **kwargs: Any):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Initialize your model/client here

    def get_batch_size(self) -> int:
        return 8  # Adjust based on your model's capabilities

    def batch_generate_response(
        self,
        queries: list[str],
        images: list[Image.Image],
        message_histories: list[Messages],
        rag_context_list: list[str],
        **kwargs,
    ) -> list[str]:
        # Your implementation here
        pass
```

#### 2. Leverage Existing Utilities

The `BaseAgent` class provides several utilities you can use:

- **`_prepare_messages()`**: Formats queries with RAG context and conversation history
- **`_build_text_content()`**: Override to customize text formatting
- **`_build_image_content()`**: Override to customize image encoding (e.g., base64)

#### 3. Choose Your Model Backend

- **For API-based models**: See `closed_model_agents.py` for examples using Anthropic, Google, and OpenAI APIs
- **For local models**: See `open_source_model_agents.py` for examples using vLLM

#### 4. Register Your Agent

Add your agent to `user_config.py`:

```python
from .my_agent import MyAgent

MODEL_KEYWORD_TO_AGENT = {
    # ... existing agents ...
    "myagent": MyAgent,
}
```

### Best Practices

- **Batch processing**: Implement efficient batching to maximize throughput
- **Error handling**: Add robust error handling for API failures or model issues
- **Context management**: Be mindful of context window limits when injecting RAG content
- **Testing**: Use `random_agent.py` as a reference for testing your agent interface

For detailed implementation examples, refer to the [Building Custom Agents](#building-custom-agents) section below.

---

## Available Agents

### Open-Source Agents

Open-source agents run locally on your hardware using **vLLM** for efficient inference. They require downloading model weights from Hugging Face and sufficient GPU resources.

| Agent | Model Family | Default Model | Requirements |
|-------|--------------|---------------|--------------|
| `LlamaAgent` | Meta Llama | `meta-llama/Llama-3.2-11B-Vision-Instruct` | GPU with vLLM |
| `PixtralAgent` | Mistral Pixtral | Pixtral models | GPU with vLLM |
| `QwenAgent` | Alibaba Qwen | Qwen-VL models | GPU with vLLM |
| `InternVLAgent` | InternVL | InternVL models | GPU with vLLM |

**Advantages:**
- Full control over model weights and inference
- No API costs or rate limits
- Works offline after initial download
- Customizable inference parameters

**Requirements:**
- NVIDIA GPU(s) with sufficient VRAM (L40s recommended)
- vLLM library installed
- Hugging Face model access (some models require authentication)

### Closed-Source Agents

Closed-source agents use external APIs to access proprietary models. They require API keys from the respective providers.

| Agent | Provider | Default Model | API Key Environment Variable |
|-------|----------|---------------|------------------------------|
| `ClaudeAgent` | Anthropic | `claude-3-7-sonnet-latest` | `ANTHROPIC_API_KEY` |
| `GeminiAgent` | Google | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| `GPTAgent` | OpenAI | `gpt-5-mini` | `OPENAI_API_KEY` |

**Advantages:**
- No GPU required
- Access to latest proprietary models
- Easy setup - just need an API key
- Automatic scaling and reliability

**Requirements:**
- Valid API key for the chosen provider
- Network connectivity to API endpoints
- API usage costs apply

---

## Quick Start

### Environment Setup

For closed-source agents, set the appropriate API key:

```bash
# Claude
export ANTHROPIC_API_KEY="your-api-key"

# Gemini
export GOOGLE_API_KEY="your-api-key"

# GPT
export OPENAI_API_KEY="your-api-key"
```

---

## Usage Examples

### Using the LlamaAgent

The `LlamaAgent` is an open-source Vision-Language Model agent that runs locally using vLLM. Here's a complete example:

```python
from PIL import Image
from agents.open_source_model_agents import LlamaAgent

# Initialize the agent with a specific model
agent = LlamaAgent(
    model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
    max_gen_len=64
)

# Load an image
image = Image.open("example.jpg")

# Prepare inputs for batch processing
queries = ["What objects are visible in this image?"]
images = [image]
message_histories = [[]]  # Empty list for single-turn conversation
rag_context_list = ["Retrieved context: This image shows a scenic landscape."]

# Generate response
responses = agent.batch_generate_response(
    queries=queries,
    images=images,
    message_histories=message_histories,
    rag_context_list=rag_context_list
)

print(responses[0])
```

#### Multi-turn Conversation Example

```python
from PIL import Image
from agents.open_source_model_agents import LlamaAgent

agent = LlamaAgent(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct")

image = Image.open("example.jpg")

# First turn
first_response = agent.batch_generate_response(
    queries=["Describe this image."],
    images=[image],
    message_histories=[[]],
    rag_context_list=[""]
)[0]

# Second turn with conversation history
history = [
    {"role": "user", "content": "Describe this image."},
    {"role": "assistant", "content": first_response}
]

second_response = agent.batch_generate_response(
    queries=["What colors are most prominent?"],
    images=[image],
    message_histories=[history],
    rag_context_list=[""]
)[0]

print(f"First response: {first_response}")
print(f"Second response: {second_response}")
```

#### Using the Agent Registry

```python
from agents.user_config import get_agent_from_model_name

# Automatically select the right agent based on model name
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
AgentClass = get_agent_from_model_name(model_name)
agent = AgentClass(model_name=model_name)
```

---

### Building Custom Agents

To create your own custom agent, inherit from `BaseAgent` and implement the required methods:

```python
from typing import Any
from PIL import Image
from agents.base_agent import BaseAgent, Messages


class MyCustomAgent(BaseAgent):
    """
    Custom RAG agent implementation.
    """

    def __init__(self, model_name: str = "my-model", **kwargs: Any):
        """
        Initialize your custom agent.

        Args:
            model_name: The name or path of the model to use.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.model_name = model_name

        # Initialize your model, tokenizer, or API client here
        self._load_model()

    def _load_model(self):
        """Load and initialize the underlying model."""
        # Your model initialization logic
        pass

    def get_batch_size(self) -> int:
        """
        Return the batch size for processing queries.

        The evaluator uses this value to determine how many queries
        to send in each batch. Valid values are 1-16.

        Returns:
            int: The batch size for this agent.
        """
        return 8  # Adjust based on your model's capabilities

    def _build_image_content(self, image: Image.Image) -> Any:
        """
        Convert a PIL Image to the format expected by your model.

        Args:
            image: The input PIL Image.

        Returns:
            The image in your model's expected format.
        """
        # Example: return base64 encoded image
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

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

        This is the main method called by the evaluator.

        Args:
            queries: List of user questions or prompts.
            images: List of PIL Image objects, one per query.
            message_histories: List of conversation histories.
                - For single-turn: Empty list []
                - For multi-turn: List of {"role": str, "content": str} dicts
            rag_context_list: Retrieved context for each query.

        Returns:
            list[str]: Generated responses, one per input query.
        """
        responses = []

        for query, image, history, rag_context in zip(
            queries, images, message_histories, rag_context_list
        ):
            # Use the built-in message preparation helper
            messages = self._prepare_messages(
                query=query,
                image=image,
                history=history,
                rag_context=rag_context
            )

            # Generate response using your model
            response = self._generate_single_response(messages, image)
            responses.append(response)

        return responses

    def _generate_single_response(self, messages: Messages, image: Image.Image) -> str:
        """
        Generate a response for a single query.

        Args:
            messages: Prepared messages including system prompt, context, and query.
            image: The associated image.

        Returns:
            str: The generated response.
        """
        # Your model inference logic here
        # Example placeholder:
        return "This is a placeholder response."
```

---

### Registering Custom Agents

After creating your custom agent, register it in `user_config.py` to make it accessible throughout the codebase:

#### Step 1: Add Import

Edit `user_config.py` and add your agent import:

```python
from .closed_model_agents import ClaudeAgent, GeminiAgent, GPTAgent
from .open_source_model_agents import InternVLAgent, LlamaAgent, PixtralAgent, QwenAgent
from .my_custom_agent import MyCustomAgent  # Add this line
```

#### Step 2: Register in MODEL_KEYWORD_TO_AGENT

Add a keyword mapping for your agent:

```python
MODEL_KEYWORD_TO_AGENT = {
    "pixtral": PixtralAgent,
    "claude": ClaudeAgent,
    "gpt": GPTAgent,
    "gemini": GeminiAgent,
    "llama": LlamaAgent,
    "internvl": InternVLAgent,
    "qwen": QwenAgent,
    "mycustom": MyCustomAgent,  # Add this line
}
```

#### Step 3: Use Your Agent

Now your agent can be used via the registry:

```python
from agents.user_config import get_agent_from_model_name

# The keyword "mycustom" will map to MyCustomAgent
agent = get_agent_from_model_name("mycustom-v1-large")
```

Or instantiate directly:

```python
from agents.my_custom_agent import MyCustomAgent

agent = MyCustomAgent(model_name="mycustom-v1-large")
responses = agent.batch_generate_response(
    queries=["What is in this image?"],
    images=[image],
    message_histories=[[]],
    rag_context_list=["Retrieved context here."]
)
```

---

## API Reference

### BaseAgent Methods

| Method | Description |
|--------|-------------|
| `get_batch_size() -> int` | Returns the batch size for query processing (1-16). |
| `batch_generate_response(queries, images, message_histories, rag_context_list) -> list[str]` | Main method for generating responses. |
| `_prepare_messages(query, image, history, rag_context) -> Messages` | Helper to prepare standardized message format. |
| `_build_text_content(text) -> Any` | Override to customize text content format. |
| `_build_image_content(image) -> Any` | Override to customize image content format. |

### Message Format

Messages follow the standard chat format:

```python
{
    "role": "user" | "assistant" | "system",
    "content": str | list[dict]
}
```

### Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `ANTHROPIC_API_KEY` | ClaudeAgent | Anthropic API authentication |
| `GOOGLE_API_KEY` | GeminiAgent | Google AI API authentication |
| `OPENAI_API_KEY` | GPTAgent | OpenAI API authentication |

---

## Contributing

When adding new agents:

1. Inherit from `BaseAgent` or an appropriate subclass
2. Implement all required abstract methods
3. Add comprehensive docstrings
4. Register in `user_config.py`
5. Add tests for your agent
