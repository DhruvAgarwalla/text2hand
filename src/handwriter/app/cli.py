"""Command-line interface."""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
