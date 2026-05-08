"""LangGraph StateGraph definition for the 3D generation agent."""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    node_feedback,
    node_gather_input,
    node_generate,
    node_save,
    node_select_format,
    node_select_llm,
    node_visualize,
)
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_generate(state: AgentState) -> str:
    if state.get("interrupted"):
        return "gather_input"
    if state.get("model_path"):
        return "save"
    # Still has retries left (last_error set, model_path None, change_description False)
    if state.get("change_description"):
        return "gather_input"
    return "generate"


def _route_after_feedback(state: AgentState) -> str:
    if state.get("user_satisfied"):
        return END
    if state.get("change_description"):
        return "gather_input"
    return "generate"


def _route_after_select_llm(state: AgentState) -> str:
    if state.get("interrupted"):
        return END
    return "select_format"


def _route_after_gather_input(state: AgentState) -> str:
    if state.get("interrupted"):
        return END
    return "generate"


def _route_after_select_format(state: AgentState) -> str:
    if state.get("interrupted"):
        return END
    return "gather_input"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("select_llm", node_select_llm)
    graph.add_node("select_format", node_select_format)
    graph.add_node("gather_input", node_gather_input)
    graph.add_node("generate", node_generate)
    graph.add_node("save", node_save)
    graph.add_node("visualize", node_visualize)
    graph.add_node("feedback", node_feedback)

    graph.add_edge(START, "select_llm")

    graph.add_conditional_edges("select_llm", _route_after_select_llm, {
        "select_format": "select_format",
        END: END,
    })
    graph.add_conditional_edges("select_format", _route_after_select_format, {
        "gather_input": "gather_input",
        END: END,
    })
    graph.add_conditional_edges("gather_input", _route_after_gather_input, {
        "generate": "generate",
        END: END,
    })
    graph.add_conditional_edges("generate", _route_after_generate, {
        "save": "save",
        "generate": "generate",
        "gather_input": "gather_input",
    })
    graph.add_edge("save", "visualize")
    graph.add_edge("visualize", "feedback")
    graph.add_conditional_edges("feedback", _route_after_feedback, {
        "gather_input": "gather_input",
        "generate": "generate",
        END: END,
    })

    return graph.compile()
