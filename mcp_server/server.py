"""MCP server exposing 3D model tools for use by Claude Desktop or other MCP clients."""
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime as dt
from pathlib import Path

from mcp.server.fastmcp import FastMCP

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def _log(msg: str) -> None:
    ts = dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] [MCP] {msg}", file=sys.stderr, flush=True)


@asynccontextmanager
async def _lifespan(server):
    _log("3DAgent Tools MCP server started")
    yield
    _log("3DAgent Tools MCP server stopped")


app = FastMCP("3DAgent Tools", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.tool()
def list_supported_formats() -> list[str]:
    """Return the list of 3D file formats supported by this agent."""
    _log("tool call: list_supported_formats()")
    with open(CONFIG_PATH) as f:
        return json.load(f)["supported_formats"]


@app.tool()
def analyze_3d_file(path: str) -> dict:
    """
    Return mesh statistics for a 3D file.

    Returns a dict with keys: vertices, faces, is_watertight, volume, bounds, format.
    """
    _log(f"tool call: analyze_3d_file(path={path!r})")
    import trimesh

    p = Path(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}

    try:
        mesh_or_scene = trimesh.load(str(p))
        if hasattr(mesh_or_scene, "geometry"):
            meshes = list(mesh_or_scene.geometry.values())
            mesh = trimesh.util.concatenate(meshes) if meshes else None
        else:
            mesh = mesh_or_scene

        if mesh is None:
            return {"error": "No geometry found in file."}

        return {
            "format": p.suffix.lstrip(".").lower(),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "bounds": mesh.bounds.tolist(),
            "extents": mesh.extents.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def validate_3d_file(path: str) -> dict:
    """
    Validate a 3D file for common issues.

    Returns a dict with keys: valid (bool), warnings (list[str]).
    """
    _log(f"tool call: validate_3d_file(path={path!r})")
    import trimesh

    p = Path(path)
    if not p.exists():
        return {"valid": False, "warnings": [f"File not found: {path}"]}

    try:
        mesh_or_scene = trimesh.load(str(p))
        if hasattr(mesh_or_scene, "geometry"):
            meshes = list(mesh_or_scene.geometry.values())
            mesh = trimesh.util.concatenate(meshes) if meshes else None
        else:
            mesh = mesh_or_scene

        if mesh is None:
            return {"valid": False, "warnings": ["No geometry."]}

        warnings: list[str] = []
        if not mesh.is_watertight:
            warnings.append("Mesh is not watertight (may cause 3D printing issues).")
        if len(mesh.faces) == 0:
            warnings.append("Mesh has no faces.")
        if len(mesh.vertices) == 0:
            warnings.append("Mesh has no vertices.")
        if hasattr(mesh, "is_winding_consistent") and not mesh.is_winding_consistent:
            warnings.append("Face winding is inconsistent.")

        return {"valid": len(warnings) == 0 or (len(mesh.faces) > 0), "warnings": warnings}
    except Exception as e:
        return {"valid": False, "warnings": [str(e)]}


@app.tool()
def convert_3d_format(input_path: str, output_format: str) -> dict:
    """
    Convert a 3D file to a different format.

    output_format should be one of: stl, obj, ply, glb, gltf, 3mf.
    Returns a dict with output_path or error.
    """
    _log(f"tool call: convert_3d_format(input={input_path!r}, format={output_format!r})")
    import trimesh

    p = Path(input_path)
    if not p.exists():
        return {"error": f"File not found: {input_path}"}

    out_path = p.with_suffix(f".{output_format.lower()}")

    try:
        mesh = trimesh.load(str(p))
        mesh.export(str(out_path))
        return {"output_path": str(out_path)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def apply_transform(
    path: str,
    scale: float = 1.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    translate_z: float = 0.0,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
) -> dict:
    """
    Apply scale, translation, and rotation to a 3D file and save the result.

    Overwrites the input file. Returns the output path or an error.
    """
    _log(f"tool call: apply_transform(path={path!r}, scale={scale}, tx={translate_x}, ty={translate_y}, tz={translate_z})")
    import numpy as np
    import trimesh

    p = Path(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}

    try:
        mesh = trimesh.load(str(p))

        if scale != 1.0:
            mesh.apply_scale(scale)

        if any([translate_x, translate_y, translate_z]):
            mesh.apply_translation([translate_x, translate_y, translate_z])

        for angle, axis in [
            (rotate_x_deg, [1, 0, 0]),
            (rotate_y_deg, [0, 1, 0]),
            (rotate_z_deg, [0, 0, 1]),
        ]:
            if angle:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(angle), axis
                )
                mesh.apply_transform(rot)

        mesh.export(str(p))
        return {"output_path": str(p)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def execute_3d_code(code: str, output_path: str) -> dict:
    """
    Execute Python code (using trimesh) to generate a 3D model.

    The variable OUTPUT_PATH is injected into scope automatically.
    Returns a dict with success (bool) and output_path or error.
    """
    _log(f"tool call: execute_3d_code(output_path={output_path!r})")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.model_executor import execute_model_code

    success, message = execute_model_code(code, output_path)
    if success:
        return {"success": True, "output_path": message}
    return {"success": False, "error": message}


# ---------------------------------------------------------------------------
# Entry point (stdio transport)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(transport="stdio")
