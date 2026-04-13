"""Typer CLI: `research run` and `research ingest`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from reflective_research.config.settings import get_settings
from reflective_research.llm.factory import get_embeddings
from reflective_research.logging_config import configure_logging
from reflective_research.retrieval.chroma_store import ingest_paths
from reflective_research.service import ResearchService

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console(stderr=True)


@app.callback()
def main() -> None:
    """Reflective deep-research agent (LangGraph)."""


@app.command("run")
def run_cmd(
    query: str = typer.Argument(..., help="Research question"),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Resume / isolate state"),
) -> None:
    """Run the research graph and print the Markdown report."""
    settings = get_settings()
    configure_logging(settings)
    svc = ResearchService(settings=settings)
    try:
        out = svc.run(query, thread_id=thread_id)
    except Exception as e:
        console.print(f"[red]Run failed:[/red] {e}")
        raise typer.Exit(code=1) from e
    finally:
        svc.shutdown()

    report = out.get("report", "")
    if out.get("errors"):
        console.print("[yellow]Warnings / errors:[/yellow]")
        for err in out["errors"]:
            console.print(f"  - {err}")
    console.print(Markdown(report or "_No report produced._"))


@app.command("serve")
def serve_cmd(
    host: str | None = typer.Option(None, "--host", help="Bind address"),
    port: int | None = typer.Option(None, "--port", help="TCP port"),
    reload: bool = typer.Option(False, "--reload", help="Dev auto-reload (needs dev install)"),
) -> None:
    """Run FastAPI (REST + SSE). Install: pip install -e ".[web]" """
    try:
        import uvicorn
    except ImportError as e:
        console.print('[red]Missing deps. Run:[/red] pip install -e ".[web]"')
        raise typer.Exit(code=1) from e
    settings = get_settings()
    configure_logging(settings)
    h = host or settings.api_host
    p = port or settings.api_port
    uvicorn.run(
        "reflective_research.api.app:app",
        host=h,
        port=p,
        reload=reload,
    )


@app.command("ingest")
def ingest_cmd(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True, help="Files or dirs"),
) -> None:
    """Chunk and index documents into Chroma (txt, md, pdf if pypdf installed)."""
    settings = get_settings()
    configure_logging(settings)
    emb = get_embeddings(settings)
    n = ingest_paths(settings, emb, paths)
    typer.echo(f"Ingested {n} chunks into {settings.chroma_persist_dir}")


if __name__ == "__main__":
    app()
