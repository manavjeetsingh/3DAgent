"""LangGraph node functions for each step of the 3D generation flow."""
import base64
import datetime
import json
import tempfile
import threading
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from agent.prompts import (
    GENERATION_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    REGENERATION_SYSTEM_PROMPT,
)
from agent.state import AgentState
from tools.model_executor import execute_model_code, extract_code
from tools.visualization import visualize_model
from utils.key_manager import get_key, save_custom_llm
from utils.llm_factory import create_llm

console = Console()

_cancel_event = threading.Event()

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Interrupt-aware input helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> Optional[str]:
    """Prompt the user. Returns None if Ctrl+D (redo signal) is pressed."""
    try:
        return Prompt.ask(prompt, default=default) if default else Prompt.ask(prompt)
    except EOFError:
        console.print("\n[yellow]Input cancelled (Ctrl+D).[/yellow]")
        return None


def _stream_llm(llm, messages: list) -> Optional[str]:
    """Stream LLM output token-by-token. Ctrl+C cancels and returns None."""
    _cancel_event.clear()
    full_response = ""
    console.print()
    try:
        for chunk in llm.stream(messages):
            if _cancel_event.is_set():
                console.print("\n[yellow]Generation cancelled.[/yellow]")
                return None
            content = chunk.content
            if isinstance(content, str) and content:
                print(content, end="", flush=True)
                full_response += content
        print()  # newline after stream
        return full_response
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted (Ctrl+C).[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Error log helper
# ---------------------------------------------------------------------------

def _save_error_log(
    code: str, error: str, description: str, attempt: int, llm_output: str = ""
) -> str:
    log_dir = Path("~/3DAgent_outputs/error_logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"error_{timestamp}_attempt{attempt}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== 3DAgent Error Log ===\n")
        f.write(f"Timestamp : {datetime.datetime.now().isoformat()}\n")
        f.write(f"Attempt   : {attempt}\n")
        f.write(f"Description:\n{description}\n\n")
        f.write(f"=== Raw LLM Output ===\n{llm_output}\n\n")
        f.write(f"=== Extracted Code ===\n{code}\n\n")
        f.write(f"=== Error ===\n{error}\n")
    return str(log_file)


# ---------------------------------------------------------------------------
# LLM connection test
# ---------------------------------------------------------------------------

def _test_llm_connection(llm_option: dict) -> bool:
    """Send a single 'hello' message to verify the API key and model work."""
    console.print(f"[cyan]Testing connection to {llm_option['name']}…[/cyan]")
    try:
        llm = create_llm(llm_option, streaming=False)
        response = llm.invoke([HumanMessage(content="Reply with the single word: OK")])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            reply = " ".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content
            ).strip()[:120]
        else:
            reply = str(content).strip()[:120]
        console.print(f"[green]✓ Connected — model replied: {reply!r}[/green]")
        return True
    except Exception as e:
        console.print(Panel(str(e), title="[red]Connection failed[/red]", border_style="red"))
        return Confirm.ask("Use this model anyway?", default=False)


# ---------------------------------------------------------------------------
# Node: select_llm
# ---------------------------------------------------------------------------

def node_select_llm(state: AgentState) -> AgentState:
    config = _load_config()
    options = config["llm_options"]

    table = Table(title="Available LLM Models", show_lines=True)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("Vision", justify="center")
    table.add_column("Key available", justify="center")

    for i, opt in enumerate(options, 1):
        has_key = "✓" if get_key(opt["key_name"]) else "✗"
        vision = "✓" if opt.get("vision") else "—"
        table.add_row(str(i), opt["name"], opt["provider"], vision, has_key)

    table.add_row(str(len(options) + 1), "[italic]Enter custom model…[/italic]", "", "", "")
    console.print(table)

    choice_str = _ask(f"Select a model [1-{len(options) + 1}]")
    if choice_str is None:
        return {**state, "interrupted": True}

    try:
        choice = int(choice_str.strip())
    except ValueError:
        console.print("[red]Invalid choice. Please enter a number.[/red]")
        return node_select_llm(state)

    if 1 <= choice <= len(options):
        selected = options[choice - 1]
    elif choice == len(options) + 1:
        selected = _handle_custom_llm(options)
        if selected is None:
            return {**state, "interrupted": True}
    else:
        console.print("[red]Choice out of range.[/red]")
        return node_select_llm(state)

    console.print(f"\n[green]Selected: {selected['name']}[/green]")

    if not _test_llm_connection(selected):
        return node_select_llm(state)

    return {**state, "selected_llm": selected}


def _handle_custom_llm(existing_options: list) -> Optional[dict]:
    name = _ask("Model display name")
    if name is None:
        return None
    provider = _ask("Provider (openai / anthropic / google / groq / openai_compatible / …)")
    if provider is None:
        return None
    model = _ask("Model ID (e.g. gpt-4o-mini)")
    if model is None:
        return None
    key_name = _ask("Environment variable name for API key (e.g. OPENAI_API_KEY)")
    if key_name is None:
        return None

    base_url = ""
    if provider == "openai_compatible":
        base_url = _ask("Base URL") or ""

    vision_str = _ask("Supports image input? (y/n)", default="n")
    vision = vision_str.lower().startswith("y")

    entry = {
        "name": name,
        "provider": provider,
        "model": model,
        "key_name": key_name,
        "vision": vision,
        "base_url": base_url,
    }

    # Test it — if it works, save permanently
    console.print("[cyan]Testing model connection…[/cyan]")
    try:
        llm = create_llm(entry, streaming=False)
        llm.invoke([HumanMessage(content="Reply with OK")])
        console.print("[green]Connection successful![/green]")
        save_custom_llm(entry)
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")
        if not Confirm.ask("Use anyway?"):
            return None

    return entry


# ---------------------------------------------------------------------------
# Node: select_format
# ---------------------------------------------------------------------------

def node_select_format(state: AgentState) -> AgentState:
    config = _load_config()
    formats = config["supported_formats"]

    table = Table(title="Supported Output Formats")
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Format")
    table.add_column("Common use")

    use_cases = {
        "stl": "3D printing (universal)",
        "obj": "Blender / general CG",
        "ply": "Point clouds / research",
        "glb": "Web / real-time (binary glTF)",
        "gltf": "Web / real-time (text glTF)",
        "3mf": "3D printing (PrusaSlicer, Bambu)",
    }

    for i, fmt in enumerate(formats, 1):
        table.add_row(str(i), fmt.upper(), use_cases.get(fmt, ""))

    console.print(table)

    choice_str = _ask(f"Select format [1-{len(formats)}]")
    if choice_str is None:
        return {**state, "interrupted": True}

    try:
        choice = int(choice_str.strip())
    except ValueError:
        console.print("[red]Invalid choice.[/red]")
        return node_select_format(state)

    if not (1 <= choice <= len(formats)):
        console.print("[red]Choice out of range.[/red]")
        return node_select_format(state)

    selected_format = formats[choice - 1]
    console.print(f"\n[green]Output format: {selected_format.upper()}[/green]")
    return {**state, "selected_format": selected_format}


# ---------------------------------------------------------------------------
# Node: gather_input
# ---------------------------------------------------------------------------

def node_gather_input(state: AgentState) -> AgentState:
    console.print(Panel(
        "Describe the 3D object you want to create.\n"
        "You may also provide optional paths to a reference image and/or an existing 3D model.\n"
        "[dim]Press Ctrl+D at any prompt to cancel and return to the previous step.[/dim]",
        title="[bold]Object Description[/bold]",
    ))

    text = _ask("Description (text)")
    if text is None:
        return {**state, "interrupted": True}

    selected_llm = state.get("selected_llm") or {}
    supports_vision = selected_llm.get("vision", False)

    image_paths: list[str] = []
    if supports_vision:
        console.print("[dim]Add reference images one by one — leave the path blank to finish.[/dim]")
        while True:
            img_input = _ask(f"  Image {len(image_paths) + 1} path", default="")
            if img_input is None:
                return {**state, "interrupted": True}
            if not img_input.strip():
                break
            p = Path(img_input.strip())
            if p.exists():
                image_paths.append(str(p))
                console.print(f"[green]  Added: {p.name}[/green]")
            else:
                console.print(f"[yellow]  Not found: {img_input}; skipping.[/yellow]")
    else:
        console.print("[dim]Selected model does not support image input.[/dim]")

    ref_model_paths: list[str] = []
    console.print("[dim]Add reference 3D models one by one (STL/OBJ/PLY) — leave blank to finish.[/dim]")
    while True:
        ref_input = _ask(f"  Model {len(ref_model_paths) + 1} path", default="")
        if ref_input is None:
            return {**state, "interrupted": True}
        if not ref_input.strip():
            break
        p = Path(ref_input.strip())
        if p.exists():
            ref_model_paths.append(str(p))
            console.print(f"[green]  Added: {p.name}[/green]")
        else:
            console.print(f"[yellow]  Not found: {ref_input}; skipping.[/yellow]")

    return {
        **state,
        "text_description": text,
        "image_paths": image_paths,
        "reference_model_paths": ref_model_paths,
        "generation_attempt": 0,
        "last_error": None,
        "generated_code": None,
        "model_path": None,
    }


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def node_generate(state: AgentState) -> AgentState:
    selected_llm = state["selected_llm"]
    fmt = state["selected_format"]
    attempt = state.get("generation_attempt", 0) + 1
    config = _load_config()
    max_retries = config.get("max_generation_retries", 3)

    console.print(Panel(
        f"[bold cyan]Generating 3D model[/bold cyan] (attempt {attempt}/{max_retries})\n"
        "[dim]Press Ctrl+C to cancel.[/dim]"
    ))

    llm = create_llm(selected_llm)

    # Build message content
    content: list = []

    text_desc = state.get("text_description", "")
    last_error = state.get("last_error")
    prev_code = state.get("generated_code")
    ref_model_paths = state.get("reference_model_paths") or []

    if last_error and prev_code:
        system_prompt = REGENERATION_SYSTEM_PROMPT.format(error=last_error)
        human_text = f"Previous code:\n```python\n{prev_code}\n```\n\nDescription: {text_desc}"
    elif prev_code and not last_error:
        # Refinement after user feedback (change_description=False path)
        feedback = state.get("last_error", "")  # reusing last_error field for feedback text
        system_prompt = REFINEMENT_SYSTEM_PROMPT.format(feedback=feedback)
        human_text = f"Previous code:\n```python\n{prev_code}\n```\n\nOriginal description: {text_desc}"
    else:
        system_prompt = GENERATION_SYSTEM_PROMPT
        human_text = text_desc

    if ref_model_paths:
        paths_list = "\n".join(f"  - {p}" for p in ref_model_paths)
        human_text += f"\n\nReference 3D models available at:\n{paths_list}"

    content.append({"type": "text", "text": human_text})

    # Attach reference images if available and model supports vision
    image_paths = state.get("image_paths") or []
    if image_paths and selected_llm.get("vision"):
        for img_path in image_paths:
            try:
                img_bytes = Path(img_path).read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                suffix = Path(img_path).suffix.lower().lstrip(".")
                mime = f"image/{suffix}" if suffix in ("png", "gif", "webp") else "image/jpeg"
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except Exception as e:
                console.print(f"[yellow]Could not attach image {img_path}: {e}[/yellow]")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content if len(content) > 1 else human_text),
    ]

    llm_output = _stream_llm(llm, messages)

    if llm_output is None:
        # User cancelled — keep current attempt count, let graph decide
        return {**state, "generation_attempt": attempt, "interrupted": True}

    code = extract_code(llm_output)

    if not code:
        message = (
            "LLM response contained no extractable Python code.\n\n"
            f"Raw LLM output ({len(llm_output)} chars):\n{llm_output or '(empty)'}"
        )
        log_path = _save_error_log("", message, text_desc, attempt, llm_output)
        console.print(f"\n[red bold]Code extraction failed (attempt {attempt}/{max_retries}):[/red bold]")
        console.print(Panel(message, title="[red]Error[/red]", border_style="red"))
        console.print(f"[dim]Full log saved to: {log_path}[/dim]")
        return {
            **state,
            "generated_code": "",
            "generation_attempt": attempt,
            "last_error": message,
            "model_path": None,
            "interrupted": False,
            **({"change_description": True} if attempt >= max_retries else {}),
        }

    # Build a temp output path
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        tmp_output = tmp.name

    console.print(f"\n[cyan]Executing generated code…[/cyan]")
    success, message = execute_model_code(code, tmp_output, timeout=config.get("generation_timeout_seconds", 120))

    if success:
        console.print(f"[green]Model generated successfully.[/green]")
        return {
            **state,
            "generated_code": code,
            "model_path": tmp_output,
            "generation_attempt": attempt,
            "last_error": None,
            "interrupted": False,
        }

    log_path = _save_error_log(code, message, text_desc, attempt, llm_output)
    console.print(f"\n[red bold]Execution failed (attempt {attempt}/{max_retries}):[/red bold]")
    console.print(Panel(message, title="[red]Error[/red]", border_style="red"))
    console.print(Panel(
        Syntax(code, "python", theme="monokai", line_numbers=True),
        title="[dim]Generated code[/dim]",
        border_style="dim",
    ))
    console.print(f"[dim]Full log saved to: {log_path}[/dim]")
    if attempt >= max_retries:
        console.print("[red]Max retries reached. Returning to input.[/red]")
        return {
            **state,
            "generated_code": code,
            "generation_attempt": attempt,
            "last_error": message,
            "model_path": None,
            "interrupted": False,
            "change_description": True,
        }

    return {
        **state,
        "generated_code": code,
        "generation_attempt": attempt,
        "last_error": message,
        "model_path": None,
        "interrupted": False,
    }


# ---------------------------------------------------------------------------
# Node: save
# ---------------------------------------------------------------------------

def node_save(state: AgentState) -> AgentState:
    tmp_path = Path(state["model_path"])
    config = _load_config()
    default_dir = Path(config.get("default_output_dir", "~/3DAgent_outputs")).expanduser()

    console.print(Panel(
        f"[bold]Save generated model[/bold]\n"
        f"Default directory: [cyan]{default_dir}[/cyan]\n"
        "[dim]Press Enter to use default, or type a custom path. Ctrl+D to skip.[/dim]"
    ))

    save_dir_input = _ask("Save directory", default=str(default_dir))
    if save_dir_input is None:
        console.print("[yellow]Skipping save; using temp file.[/yellow]")
        return state

    save_dir = Path(save_dir_input.strip()).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    fmt = state["selected_format"]
    existing = list(save_dir.glob(f"model_*.{fmt}"))
    index = len(existing) + 1
    dest = save_dir / f"model_{index:03d}.{fmt}"

    tmp_path.rename(dest)
    console.print(f"[green]Saved to: {dest}[/green]")
    return {**state, "model_path": str(dest), "save_directory": str(save_dir)}


# ---------------------------------------------------------------------------
# Node: visualize
# ---------------------------------------------------------------------------

def node_visualize(state: AgentState) -> AgentState:
    model_path = state.get("model_path")
    if not model_path:
        console.print("[yellow]No model path available; skipping visualization.[/yellow]")
        return state

    console.print(Panel("[bold cyan]Visualization[/bold cyan]\nOpening 2D and 3D views…"))
    try:
        visualize_model(model_path)
    except KeyboardInterrupt:
        console.print("[yellow]Visualization closed.[/yellow]")
    return state


# ---------------------------------------------------------------------------
# Node: feedback
# ---------------------------------------------------------------------------

def node_feedback(state: AgentState) -> AgentState:
    console.print(Panel(
        "[bold]Is this the object you wanted?[/bold]\n"
        "  [green]y[/green] — yes, done\n"
        "  [yellow]r[/yellow] — regenerate with same description\n"
        "  [cyan]m[/cyan] — modify description and regenerate\n"
        "  [red]q[/red] — quit"
    ))

    choice = _ask("Choice [y/r/m/q]")
    if choice is None:
        choice = "r"

    choice = choice.strip().lower()

    if choice == "y":
        console.print("[green]Great! Model saved. Goodbye.[/green]")
        return {**state, "user_satisfied": True, "change_description": False}

    if choice == "r":
        feedback_text = _ask("Describe what to improve (optional)", default="") or ""
        return {
            **state,
            "user_satisfied": False,
            "change_description": False,
            "last_error": feedback_text,
            "generation_attempt": 0,
        }

    if choice == "m":
        return {**state, "user_satisfied": False, "change_description": True, "generation_attempt": 0}

    if choice == "q":
        console.print("[dim]Quitting.[/dim]")
        return {**state, "user_satisfied": True, "change_description": False}

    console.print("[red]Unrecognised choice.[/red]")
    return node_feedback(state)
