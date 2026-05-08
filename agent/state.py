from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # LLM selection
    selected_llm: Optional[dict]          # full entry from config.json llm_options

    # Format
    selected_format: Optional[str]        # e.g. "stl"

    # User inputs
    text_description: Optional[str]
    image_paths: Optional[list[str]]          # paths to reference images
    reference_model_paths: Optional[list[str]] # paths to existing 3D files

    # Generation
    generated_code: Optional[str]
    last_error: Optional[str]
    generation_attempt: int               # retry counter

    # Output
    model_path: Optional[str]            # final saved file path
    save_directory: Optional[str]

    # Feedback
    user_satisfied: Optional[bool]
    change_description: Optional[bool]   # True → re-gather input, False → regenerate

    # Control
    interrupted: bool
