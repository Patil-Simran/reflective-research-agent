"""Environment-driven configuration (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path | None:
    """Locate install root (directory with this package under src/)."""
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() and (d / "src" / "reflective_research").is_dir():
            return d
    return None


def _discover_env_files() -> tuple[str, ...]:
    """
    Load .env from the repo first, then cwd — so OLLAMA_BASE_URL works when
    `research serve` is started outside the project directory.
    Later files override earlier ones for duplicate keys.
    """
    paths: list[Path] = []
    root = _find_repo_root()
    if root is not None:
        repo_env = root / ".env"
        if repo_env.is_file():
            paths.append(repo_env)
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        if not paths or cwd_env.resolve() != paths[0].resolve():
            paths.append(cwd_env)
    return tuple(str(p) for p in paths) if paths else (".env",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_discover_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    #: Chat model for plan + reflect. For deeper reports set OLLAMA_WRITER_MODEL to a larger tag.
    ollama_model: str = Field(default="llama3.2:latest")
    #: Optional larger/better model for synthesize + revise only (empty = same as OLLAMA_MODEL).
    ollama_writer_model: str = Field(default="")
    #: Second Ollama tag for verify step; empty string = reuse OLLAMA_MODEL (separate ChatOllama, temp=0).
    ollama_verifier_model: str = Field(default="")
    #: Max tokens for writer synthesis (Ollama `num_predict`). Higher = longer reports.
    ollama_num_predict: int = Field(default=6144, ge=512, le=131072)
    #: Context window for the chat model (Ollama `num_ctx`). Must fit model limits.
    ollama_num_ctx: int = Field(default=16384, ge=2048, le=131072)

    embedding_provider: Literal["ollama", "huggingface"] = Field(default="huggingface")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    hf_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    chroma_collection: str = Field(default="research_corpus")

    max_reflection_iterations: int = Field(default=5, ge=1, le=20)
    #: After synthesize: verify + optional revise loops (0 = verify only, no revise).
    max_verification_revisions: int = Field(default=2, ge=0, le=5)
    verification_enabled: bool = Field(default=True)
    search_max_results: int = Field(default=7, ge=1, le=20)
    #: Optional Brave Search API (https://brave.com/search/api/) — strong web results without DDG.
    brave_search_api_key: str = Field(default="")
    search_ddg_instant_enabled: bool = Field(default=True)
    #: duckduckgo_search package (text/news/images); often breaks when upstream changes — off by default.
    search_duckduckgo_package_enabled: bool = Field(default=False)
    search_semantic_scholar_enabled: bool = Field(default=True)
    search_semantic_scholar_max_results: int = Field(default=6, ge=0, le=15)
    search_crossref_enabled: bool = Field(default=True)
    search_crossref_max_results: int = Field(default=5, ge=0, le=15)
    #: Crossref polite-pool etiquette (https://github.com/Crossref/rest-api-doc)
    crossref_mailto: str = Field(default="mailto:research-agent@localhost")
    search_arxiv_enabled: bool = Field(default=True)
    search_arxiv_max_results: int = Field(default=6, ge=0, le=15)
    search_news_fallback_enabled: bool = Field(default=True)
    search_news_max_results: int = Field(default=6, ge=0, le=15)
    image_search_enabled: bool = Field(default=True)
    image_search_max_per_query: int = Field(default=4, ge=0, le=12)
    #: Chunks retrieved per RAG plan step.
    rag_top_k: int = Field(default=6, ge=1, le=20)
    #: Concurrent plan steps per gather round (search + RAG steps run in parallel).
    gather_parallelism: int = Field(default=4, ge=1, le=16)
    url_fetch_enabled: bool = Field(default=True)
    url_fetch_per_search_step: int = Field(default=2, ge=0, le=8)
    url_fetch_max_bytes: int = Field(default=1_500_000, ge=10_000, le=10_000_000)
    url_fetch_timeout_s: float = Field(default=12.0, ge=2.0, le=60.0)
    evidence_rerank_enabled: bool = Field(default=True)
    #: Add a small score bonus for Wikipedia, arXiv, .edu, etc. (open deep-research style tie-break).
    evidence_authority_bonus_enabled: bool = Field(default=True)
    evidence_substantive_overlap_fraction: float = Field(default=0.12, ge=0.02, le=0.5)
    evidence_substantive_max_required_hits: int = Field(default=6, ge=1, le=20)
    search_block_commerce_hosts: bool = Field(default=True)
    #: When **False** (default), Zhihu / Baidu Zhidao / similar CN UGC Q&A are **never** ingested
    #: or returned in the API evidence list. Set **True** only for Chinese-first research.
    search_allow_chinese_qa_mirrors: bool = Field(default=False)
    #: Deprecated — host blocking uses ``search_allow_chinese_qa_mirrors`` only. Kept so old
    #: ``SEARCH_PREFER_ENGLISH_SOURCES`` in .env does not error.
    search_prefer_english_sources: bool = Field(default=True)
    #: Drop dictionary / “define word” mirrors and a few brand landers that match generic tokens (e.g. CASE).
    search_block_glossary_spam_hosts: bool = Field(default=True)
    evidence_brief_enabled: bool = Field(default=True)
    request_timeout_s: float = Field(default=120.0, ge=5.0)

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # API / frontend (comma-separated origins)
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    #: API/UI ``evidence`` list: only rows cited as [n] in the report (when citations exist).
    api_evidence_cited_only: bool = Field(default=True)
    #: Dev convenience: allow any localhost port (5174, etc.) without editing CORS_ORIGINS.
    cors_allow_localhost_regex: bool = Field(default=True)

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def _normalize_ollama_base_url(cls, v: object) -> str:
        """
        LangChain passes this into ``ollama.Client(host=...)``. If the URL has no
        scheme, ``parse_url_with_auth`` returns None and the client ignores
        OLLAMA_BASE_URL and uses env ``OLLAMA_HOST`` only — looks like your setting
        is "ignored". Always produce a full URL with scheme and host.
        """
        if v is None:
            return "http://127.0.0.1:11434"
        s = str(v).strip().strip('"').strip("'")
        if not s:
            return "http://127.0.0.1:11434"
        if not s.lower().startswith(("http://", "https://")):
            s = f"http://{s}"
        parsed = urlparse(s)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"Invalid OLLAMA_BASE_URL {v!r}. Example: http://127.0.0.1:11434"
            )
        # No path/query — Ollama API is at the origin
        return urlunparse((parsed.scheme.lower(), parsed.netloc, "", "", "", ""))

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (reload process to pick up .env changes)."""
    return Settings()
