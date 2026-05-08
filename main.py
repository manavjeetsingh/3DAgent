#!/usr/bin/env python3
"""
3DAgent — conversational agent for generating 3D models.

Usage:
    python main.py

Keyboard shortcuts during a session:
    Ctrl+C  — cancel current LLM generation, return to previous step
    Ctrl+D  — at any text prompt, cancel that prompt and return to previous step
"""
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Make local packages importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from agent.graph import build_graph
from agent.state import AgentState

console = Console()

BANNER = """
[bold cyan]  _____ ____      _                    _   [/bold cyan]
[bold cyan] |___ /|  _ \\    / \\   __ _  ___ _ __ | |_ [/bold cyan]
[bold cyan]   |_ \\| | | |  / _ \\ / _` |/ _ \\ '_ \\| __|[/bold cyan]
[bold cyan]  ___) | |_| | / ___ \\ (_| |  __/ | | | |_ [/bold cyan]
[bold cyan] |____/|____/ /_/   \\_\\__, |\\___|_| |_|\\__|[/bold cyan]
[bold cyan]                       |___/                 [/bold cyan]

[dim]Conversational agent for generating 3D models via LLMs[/dim]
[dim]Press Ctrl+C during generation to cancel · Ctrl+D at prompts to go back[/dim]
"""


def main() -> None:
    console.print(BANNER)
    console.print(Panel(
        "[bold]Welcome to 3DAgent[/bold]\n\n"
        "This tool will guide you through selecting a language model,\n"
        "choosing an output format, describing your 3D object, and\n"
        "generating, previewing, and saving the model file.",
        border_style="cyan",
    ))

    graph = build_graph()
    initial_state: AgentState = {
        "selected_llm": None,
        "selected_format": None,
        "text_description": None,
        "image_paths": [],
        "reference_model_paths": [],
        "generated_code": None,
        "last_error": None,
        "generation_attempt": 0,
        "model_path": None,
        "save_directory": None,
        "user_satisfied": None,
        "change_description": False,
        "interrupted": False,
    }

    try:
        final_state = graph.invoke(initial_state)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Session interrupted. Goodbye.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        raise

    model_path = final_state.get("model_path")
    if model_path:
        console.print(f"\n[bold green]Final model: {model_path}[/bold green]")
    console.print("[dim]Session complete.[/dim]")


if __name__ == "__main__":
    main()
