"""Command-line interface."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from handwriter import __version__
from handwriter.config import DEFAULT_DPI, a4_size_px

app = typer.Typer(help="Generate documents in realistic handwriting.", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"handwriter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Handwriting Replicator."""


@app.command()
def info() -> None:
    """Show basic configuration."""
    w, h = a4_size_px()
    typer.echo(f"handwriter {__version__}")
    typer.echo(f"Page: A4, {w} x {h} px at {DEFAULT_DPI} DPI")


@app.command()
def styles() -> None:
    """List handwriting styles, page templates and ink colours."""
    from handwriter.render.pen import INK_COLORS
    from handwriter.styles import available_styles
    from handwriter.templates.pages import TEMPLATES

    typer.echo("Styles:")
    for label, key in available_styles():
        typer.echo(f"  {key:<24} {label}")
    typer.echo("Pages:")
    for key, t in TEMPLATES.items():
        typer.echo(f"  {key:<24} {t.label}")
    typer.echo("Inks: " + ", ".join(INK_COLORS))


@app.command()
def render(
    input: str = typer.Argument(..., help="Text file to write out, or '-' for stdin."),
    output: Path = typer.Option(Path("handwritten.pdf"), "-o", "--output",
                                help="Output file: .pdf, .docx or .png"),
    style: str = typer.Option("print", help="Style key (see `handwriter styles`)."),
    page: str = typer.Option("ruled-college", help="Page template key."),
    background: Path | None = typer.Option(None, help="Scanned page (PDF/image) to write on."),
    messiness: float = typer.Option(1.0, help="0 = very neat ... 2.5 = very messy."),
    size: float = typer.Option(0.40, help="Letter x-height as a fraction of line spacing."),
    pen: float = typer.Option(0.38, help="Pen width in mm."),
    ink: str = typer.Option("blue", help="Ink colour."),
    slant: float | None = typer.Option(None, help="Extra slant in degrees (+ leans right)."),
    seed: int | None = typer.Option(None, help="Random seed for reproducible output."),
    dpi: int = typer.Option(DEFAULT_DPI, help="Output resolution."),
) -> None:
    """Write a text file out in handwriting."""
    from handwriter.export.writers import save
    from handwriter.pipeline import RenderSettings, render_pages

    text = sys.stdin.read() if input == "-" else Path(input).read_text(encoding="utf-8")
    settings = RenderSettings(style=style, page=page, messiness=messiness, size=size,
                              pen_mm=pen, ink=ink, slant_deg=slant, seed=seed, dpi=dpi)
    factory = None
    if background is not None:
        from handwriter.templates.scanned import scanned_page_factory

        factory = scanned_page_factory(background, settings)
    t0 = time.time()
    result = render_pages(text, settings, page_factory=factory)
    save(result.images, output, dpi=dpi)
    typer.echo(f"Wrote {len(result.images)} page(s) to {output} in {time.time() - t0:.1f}s "
               f"(seed {result.seed})")
    if result.missing:
        typer.echo(f"Warning: no glyph for: {' '.join(result.missing)}", err=True)


@app.command()
def web(
    port: int = typer.Option(7860, help="Port for the local web app."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Start the local web app."""
    from handwriter.app.web import launch

    launch(port=port, share=share)


if __name__ == "__main__":
    app()
