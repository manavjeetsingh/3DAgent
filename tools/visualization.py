"""2D and interactive 3D visualization of generated 3D models."""
from pathlib import Path

from rich.console import Console

console = Console()


def visualize_model(model_path: str) -> None:
    """Show interactive 3D view (pyvista) and 2D orthographic views (matplotlib)."""
    try:
        import trimesh
    except ImportError:
        console.print("[red]trimesh not installed. Run: pip install trimesh[/red]")
        return

    path = Path(model_path)
    if not path.exists():
        console.print(f"[red]File not found: {model_path}[/red]")
        return

    console.print(f"\n[cyan]Loading model: {path.name}[/cyan]")
    try:
        mesh_or_scene = trimesh.load(str(path))
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    # Resolve Scene → single Trimesh
    if hasattr(mesh_or_scene, "geometry"):
        meshes = list(mesh_or_scene.geometry.values())
        if not meshes:
            console.print("[red]Scene contains no geometry.[/red]")
            return
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = mesh_or_scene

    console.print(
        f"  Vertices: {len(mesh.vertices):,}  |  "
        f"Faces: {len(mesh.faces):,}  |  "
        f"Volume: {mesh.volume:.4f}"
    )

    _show_2d_views(mesh)
    _show_3d_interactive(model_path)


def _show_2d_views(mesh) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        console.print("[yellow]matplotlib not available; skipping 2D views.[/yellow]")
        return

    verts = mesh.vertices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Orthographic Views", fontsize=14)

    projections = [
        ("Top (XY)", verts[:, 0], verts[:, 1]),
        ("Front (XZ)", verts[:, 0], verts[:, 2]),
        ("Side (YZ)", verts[:, 1], verts[:, 2]),
    ]

    for ax, (title, x, y) in zip(axes, projections):
        ax.scatter(x, y, s=0.5, alpha=0.4, color="steelblue")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show(block=False)
    console.print("[green]2D views displayed (close window to continue).[/green]")


def _show_3d_interactive(model_path: str) -> None:
    try:
        import pyvista as pv
    except ImportError:
        console.print("[yellow]pyvista not available; skipping interactive 3D view.[/yellow]")
        _fallback_3d_trimesh(model_path)
        return

    try:
        mesh = pv.read(model_path)
        plotter = pv.Plotter(title="3DAgent — Interactive View")
        plotter.add_mesh(
            mesh,
            show_edges=True,
            edge_color="black",
            color="lightsteelblue",
            opacity=0.9,
        )
        plotter.add_axes()
        plotter.show_grid()
        console.print(
            "[green]Interactive 3D view open — rotate with left-click drag, "
            "zoom with scroll, close window to continue.[/green]"
        )
        plotter.show()
    except Exception as e:
        console.print(f"[yellow]pyvista render failed ({e}); falling back to trimesh viewer.[/yellow]")
        _fallback_3d_trimesh(model_path)


def _fallback_3d_trimesh(model_path: str) -> None:
    try:
        import trimesh
        mesh_or_scene = trimesh.load(model_path)
        mesh_or_scene.show()
    except Exception as e:
        console.print(f"[red]Could not show 3D view: {e}[/red]")
