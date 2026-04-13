"""LLM / embedding factories — fixed free stack: Ollama chat + HF or Ollama embeddings."""

from __future__ import annotations

import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from reflective_research.config.settings import Settings

log = logging.getLogger(__name__)


def get_chat_model(settings: Settings) -> BaseChatModel:
    """LLM is always local Ollama (no external inference API keys)."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.25,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        timeout=settings.request_timeout_s,
    )


def get_writer_chat_model(settings: Settings) -> BaseChatModel:
    """
    Long-form synthesis + revision. Use OLLAMA_WRITER_MODEL for a larger instruct model
    while keeping OLLAMA_MODEL smaller for plan/reflect.
    """
    wm = (settings.ollama_writer_model or "").strip()
    model = wm or settings.ollama_model
    if wm:
        log.info("Writer model: %s (plan/reflect: %s)", model, settings.ollama_model)
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.25,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        timeout=settings.request_timeout_s,
    )


def get_verifier_chat_model(settings: Settings) -> BaseChatModel:
    """
    Independent reviewer: lower temperature, optional different Ollama tag for cross-model check.
    If OLLAMA_VERIFIER_MODEL is empty, uses OLLAMA_MODEL (still a separate client instance).
    """
    vm = (settings.ollama_verifier_model or "").strip()
    model = vm or settings.ollama_model
    if vm and vm != settings.ollama_model:
        log.info("Verifier cross-model: %s (writer: %s)", model, settings.ollama_model)
    else:
        log.info("Verifier same tag as writer (%s), cold temperature for 2nd pass", model)
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=min(settings.ollama_num_ctx, 8192),
        num_predict=min(settings.ollama_num_predict, 4096),
        timeout=settings.request_timeout_s,
    )


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    if settings.embedding_provider == "huggingface":
        log.info("Embeddings: HuggingFace %s (local CPU)", settings.hf_embedding_model)
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
