import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()

KEYS_FILE = Path(__file__).parent.parent / "keys.json"


def _load_keys() -> dict:
    if not KEYS_FILE.exists():
        return {}
    with open(KEYS_FILE) as f:
        return json.load(f)


def _save_keys(keys: dict) -> None:
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def get_key(key_name: str) -> Optional[str]:
    keys = _load_keys()
    value = keys.get(key_name) or os.environ.get(key_name)
    return value if value else None


def prompt_and_save_key(key_name: str, provider_name: str) -> str:
    console.print(f"\n[yellow]No API key found for {provider_name} ({key_name}).[/yellow]")
    key = Prompt.ask(f"Enter your {provider_name} API key", password=True)
    key = key.strip()
    if not key:
        raise ValueError(f"API key for {provider_name} cannot be empty.")
    keys = _load_keys()
    keys[key_name] = key
    _save_keys(keys)
    console.print(f"[green]Key saved to keys.json[/green]")
    return key


def get_or_prompt_key(key_name: str, provider_name: str) -> str:
    key = get_key(key_name)
    if key:
        return key
    return prompt_and_save_key(key_name, provider_name)


def save_custom_llm(llm_entry: dict) -> None:
    """Persist a user-defined LLM to config.json."""
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    existing_names = {e["name"] for e in config["llm_options"]}
    if llm_entry["name"] not in existing_names:
        config["llm_options"].append(llm_entry)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Added '{llm_entry['name']}' to LLM options.[/green]")
