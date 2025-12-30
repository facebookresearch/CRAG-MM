import hashlib
import io
import itertools
import json
import os
import urllib.parse
from typing import Any

import pandas as pd
import requests
from PIL import Image
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def ensure_crag_cache_dir_is_configured():
    """
    Ensure the cache directory for CRAG images exists and is properly configured.

    This function:
    1. Checks if CRAG_CACHE_DIR environment variable is set
    2. If not set, uses platform-appropriate default cache location
    3. Creates the directory if it doesn't exist
    4. Returns the path to the cache directory

    Returns:
        str: Path to the cache directory
    """
    # First check if user has explicitly set a cache directory
    cache_dir = os.environ.get("CRAG_CACHE_DIR")

    if not cache_dir:
        # Use platform-specific default locations if not explicitly set
        if os.name == "nt":  # Windows
            cache_home = os.environ.get(
                "LOCALAPPDATA", os.path.expanduser("~/AppData/Local")
            )
        else:  # Unix/Linux/Mac
            cache_home = os.environ.get(
                "XDG_CACHE_HOME", os.path.expanduser("~/.cache")
            )

        cache_dir = os.path.join(cache_home, "cragmm_images_cache")

        # Print info message only the first time
        if not hasattr(ensure_crag_cache_dir_is_configured, "_cache_location_shown"):
            print(f"Caching downloaded images in {cache_dir}")
            print(
                "You can override this by setting the CRAG_CACHE_DIR environment"
                " variable."
            )
            ensure_crag_cache_dir_is_configured._cache_location_shown = True

    # Create the directory if it doesn't exist
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create cache directory {cache_dir}: {e}")
            # Fall back to a temporary directory if we can't create the default
            import tempfile

            cache_dir = os.path.join(tempfile.gettempdir(), "cragmm_images_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using fallback cache directory: {cache_dir}")

    return cache_dir


def download_image_url(image_url):
    """Downloads image from URL and saves it to the cache directory with a deterministic
    name. Returns local path if successful, raises Exception otherwise.

    Args:
        image_url: URL of the image to download

    Returns:
        str: Local path to the downloaded or cached image

    Raises:
        Exception: If the image couldn't be downloaded or is invalid
    """
    cache_dir = ensure_crag_cache_dir_is_configured()

    # Create cache directory if it doesn't exist (redundant but keeps backward
    # compatibility)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Create a deterministic filename based on the URL
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        file_extension = os.path.splitext(image_url.split("?")[0])[1] or ".jpg"
        local_filename = f"{url_hash}{file_extension}"
        local_path = os.path.join(cache_dir, local_filename)

        # If the file already exists in cache, validate and return it
        if os.path.exists(local_path):
            if _is_valid_image(local_path):
                print(f"Using cached image from {local_path}")
                return local_path
            else:
                print(f"Cached image is invalid, re-downloading: {local_path}")
                # Continue with download as the cached file is invalid

        # Download the image
        headers = {"User-Agent": "CRAGBot/v0.0.1"}
        response = requests.get(image_url, stream=True, timeout=10, headers=headers)
        response.raise_for_status()

        # Save the image to a temporary file first
        temp_path = f"{local_path}.temp"
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate the downloaded image
        if _is_valid_image(temp_path):
            # Move to final location if valid
            os.replace(temp_path, local_path)
            print(f"Downloaded and validated image_url to {local_path}")
            return local_path
        else:
            # Remove invalid image
            os.remove(temp_path)
            print(f"Downloaded image is not valid from URL: {image_url}")
            raise Exception(f"Downloaded image is not valid from URL {image_url}")

    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        raise Exception(f"Error downloading image from {image_url}: {e}")


def _is_valid_image(file_path):
    """Check if the file is a valid image using PIL.

    Args:
        file_path: Path to the image file

    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            # Verify the image by loading it
            img.verify()

            # Additional check by accessing image properties
            width, height = img.size
            if width <= 0 or height <= 0:
                return False

            return True
    except Exception as e:
        print(f"Invalid image file {file_path}: {e}")
        return False


def is_url(url):
    """Check if the URL is a valid image URL."""
    try:
        result = urllib.parse.urlparse(url)
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


def resize_image_if_needed(
    image: Image.Image, max_size_mb: float = 5.0, max_dimension_px: int = 8000
) -> Image.Image:
    """
    Resize an image if it exceeds the allowed file size or dimension size.
    Maintains aspect ratio while shrinking.

    Args:
        image (Image.Image): The input PIL Image.
        max_size_mb (float): Max allowed size in megabytes.
        max_dimension_px (int): Max allowed width/height in pixels.

    Returns:
        Image.Image: The resized (or original) image.
    """
    max_bytes = max_size_mb * 1024 * 1024

    def get_image_size_bytes(img: Image.Image, format="JPEG", quality=85) -> int:
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality, optimize=True)
        return buf.tell()

    # --- Step 1: shrink if dimension exceeds max_dimension_px ---
    width, height = image.size
    if max(width, height) > max_dimension_px:
        scale = max_dimension_px / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    # --- Step 2: shrink if file size exceeds max_size_mb ---
    size_bytes = get_image_size_bytes(image)
    while size_bytes > max_bytes and (image.size[0] > 100 and image.size[1] > 100):
        # shrink by 90%
        new_size = (int(image.size[0] * 0.9), int(image.size[1] * 0.9))
        image = image.resize(new_size, Image.LANCZOS)
        size_bytes = get_image_size_bytes(image)

    return image


def display_results(
    console: Console,
    turn_evaluation_results_df: pd.DataFrame,
    scores_dictionary: dict[str, Any],
    display_conversations: int = 3,
    is_ego: bool = False,
    is_multi_turn: bool = False,
) -> None:
    """Display evaluation results in a formatted way"""

    ego_string = "Ego Only" if is_ego else "All"
    title = f"Evaluation Results :: {ego_string}"

    # Create metrics table
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value")

    metrics_table.add_row(
        "Total conversations", str(turn_evaluation_results_df["session_id"].nunique())
    )
    metrics_table.add_row(
        "Total turns", str(turn_evaluation_results_df["interaction_id"].nunique())
    )
    metrics_table.add_row(
        "Avg MT conversation score",
        "{:.2f}".format(scores_dictionary["mean_multi_turn_conversation_score"]),
    )
    metrics_table.add_row("Exact matches", str(scores_dictionary["correct_exact"]))
    metrics_table.add_row('"I don\'t know" responses', str(scores_dictionary["miss"]))
    metrics_table.add_row(
        "Hallucinated responses", str(scores_dictionary["hallucination"])
    )
    metrics_table.add_row("Exact accuracy", f"{scores_dictionary['exact_match']:.2%}")
    metrics_table.add_row("Accuracy", f"{scores_dictionary['accuracy']:.2%}")
    metrics_table.add_row("Missing rate", f"{scores_dictionary['missing']:.2%}")
    metrics_table.add_row(
        "Hallucination rate", f"{scores_dictionary['hallucination_rate']:.2%}"
    )
    metrics_table.add_row(
        "Truthfulness score", f"{scores_dictionary['truthfulness_score']:.4f}"
    )

    # Create a list of renderables to display in the panel
    renderables = [metrics_table]

    # Add sample conversation tables if requested
    if display_conversations > 0:

        def _init_conversation_table(title: str):
            table = Table(title=title)
            if is_multi_turn:
                table.add_column("Turn", style="dim")
            else:
                table.add_column("interaction_id", style="dim")
            table.add_column("Query", style="dim")
            table.add_column("Agent Response", style="dim")
            table.add_column("Ground Truth", style="dim")
            table.add_column("API Response", style="dim")
            table.add_column("Evaluation Result", style="dim")
            return table

        def _get_status_style_and_text(row: pd.Series):
            if row["is_exact_match"]:
                return "green", "[green]EXACT MATCH[/green]"
            elif row["is_semantically_correct"]:
                return "green", "[green]SEMANTICALLY CORRECT[/green]"
            elif row["is_miss"]:
                return "yellow", "[yellow]I DON'T KNOW[/yellow]"
            else:
                return "red", "[red]INCORRECT[/red]"

        # Add section header for sample results
        renderables.append(Text("\nSample Evaluation Results", style="bold cyan"))

        if is_multi_turn:
            conversation_groups = turn_evaluation_results_df.groupby("session_id")
            for session_id, group_df in itertools.islice(
                conversation_groups, display_conversations
            ):
                table = _init_conversation_table(f"Session ID: {session_id}")
                for idx, row in group_df.iterrows():
                    status_style, status_text = _get_status_style_and_text(row)
                    table.add_row(
                        f"{row['turn_idx']}",
                        f"[bold cyan]{row['query']}[/bold cyan]",
                        f"[bold yellow]{row['agent_response']}[/bold yellow]",
                        f"[bold green]{row['ground_truth']}[/bold green]",
                        f"[bold blue]{str(row['api_response'])[:100]}[/bold blue]",
                        f"[bold {status_style}]{status_text}[/bold {status_style}]",
                    )
                renderables.append(table)
        else:
            table = _init_conversation_table("Evaluation Results")
            for idx, row in itertools.islice(
                turn_evaluation_results_df.iterrows(), display_conversations
            ):
                status_style, status_text = _get_status_style_and_text(row)
                table.add_row(
                    f"{row['interaction_id'][:5]}",
                    f"[bold cyan]{row['query']}[/bold cyan]",
                    f"[bold yellow]{row['agent_response']}[/bold yellow]",
                    f"[bold green]{row['ground_truth']}[/bold green]",
                    f"[bold blue]{str(row['api_response'])[:100]}[/bold blue]",
                    f"[bold {status_style}]{status_text}[/bold {status_style}]",
                )
            renderables.append(table)

    # Display all tables in a single panel using Group to combine renderables
    group = Group(*renderables)
    panel = Panel(
        group, title=f"[bold]{title}[/bold]", border_style="blue", padding=(1, 2)
    )
    console.print(panel)


def parse_jsonl_to_dict(file_path):
    """
    Read a .jsonl file and parse it into a dictionary.

    Args:
        file_path (str): Path to the .jsonl file

    Returns:
        dict: Dictionary with interaction_id as keys and full_query as values
    """
    result_dict = {}

    if file_path is None:
        return result_dict

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        interaction_id = data.get("interaction_id")
                        full_query = data.get("agent_response")

                        if interaction_id is not None and full_query is not None:
                            try:
                                full_query_parsed = json.loads(full_query)
                                result_dict[interaction_id] = full_query_parsed[
                                    "updated_question"
                                ]
                            except Exception:
                                result_dict[interaction_id] = data.get("query")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error: {e}")
                        continue
    except Exception:
        pass

    return result_dict
