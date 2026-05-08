"""Execute LLM-generated Python code to produce 3D model files."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_code(llm_output: str) -> str:
    """Pull the first Python code block from LLM markdown output."""
    # Handles optional language tag, trailing spaces, and CRLF line endings
    pattern = r"```(?:python|Python)?\s*\r?\n(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fences found — return raw text only if it looks like actual code
    stripped = llm_output.strip()
    if stripped.startswith(("import ", "from ", "#", "def ", "class ", "OUTPUT_PATH")):
        return stripped
    return ""


def execute_model_code(code: str, output_path: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Write code to a temp file and execute it.

    The generated code is expected to save its output to OUTPUT_PATH (injected
    via environment variable) and print "SUCCESS: <path>" on completion.

    Returns (success, message).
    """
    # Inject the output path as the first line so LLM code can use it
    preamble = f'OUTPUT_PATH = r"{output_path}"\n\n'
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        Path(tmp_path).unlink(missing_ok=True)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            parts = []
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n\n".join(parts) or "(no output)"

        output_file = Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            parts = [f"Code exited cleanly but {output_path} was not created."]
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            return False, "\n\n".join(parts)

        return True, output_path

    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return False, f"Code execution timed out after {timeout}s."
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        return False, str(e)
